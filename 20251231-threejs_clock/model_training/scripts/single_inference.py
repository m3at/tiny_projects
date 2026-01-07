"""Single image inference script for clock reading model."""

import argparse
import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse

import huggingface_hub
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

from src.basic_transform import from_sincos

# from src.models import ClockModel, DinoBilinear
from src.models import ClockModel, DinoV3Backbone
from src.settings import p_env


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL."""
    from io import BytesIO

    import requests

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image(path_or_url: str) -> Image.Image:
    """Load image from local path or URL."""
    # Check if it's a URL
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        return load_image_from_url(path_or_url)
    else:
        # Local file
        return Image.open(path_or_url).convert("RGB")


def preprocess_image(image: Image.Image, target_dim: int = 448, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Preprocess image for model input."""
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, scale=False),
            transforms.Resize(target_dim),
            transforms.CenterCrop((target_dim, target_dim)),
            transforms.ToDtype(dtype, scale=True),
        ]
    )
    return transform(image)


def sincos_to_time(sin_h: float, cos_h: float, sin_m: float, cos_m: float) -> tuple[int, int]:
    """Convert sin/cos predictions to hour and minute."""
    hour_float = from_sincos(sin_h, cos_h)
    # minute_float = from_sincos(sin_m, cos_m)
    _ = from_sincos(sin_m, cos_m)

    # Convert to actual time
    # hour_float is in [0, 1) over 12 hours
    # minute_float is in [0, 1) over 60 minutes
    tot_seconds = int(round(hour_float * 43200)) % 43200
    hour = (tot_seconds // 3600) % 12
    minute = (tot_seconds % 3600) // 60

    return hour, minute


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single clock image")
    parser.add_argument("image", help="Path to image file or URL")
    parser.add_argument(
        "--checkpoint",
        default="./results/checkpoint.safetensors",
        help="Path to model checkpoint (default: ./results/checkpoint.safetensors)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on (cuda/mps/cpu, default: auto-detect)",
    )
    args = parser.parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # RuntimeError: MPS: Unsupported Border padding mode
            # device = torch.device("mps")
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Load image
    print(f"Loading image from: {args.image}")
    try:
        image = load_image(args.image)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    # Preprocess
    print("Preprocessing image...")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Load model
    print(f"Loading model from: {checkpoint_path}")

    # Suppress xFormers warnings from DINOv2
    warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

    # Load DINOv2 backbone
    # backbone_original = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=False)
    # backbone = DinoBilinear(backbone_original)
    # Load DINOv3 backbone
    _k = "vitl16"
    model_id = f"facebook/dinov3-{_k}-pretrain-lvd1689m"
    huggingface_hub.login(p_env.HF_TOKEN.get_secret_value())
    # dinov3 = AutoModel.from_pretrained(model_id).eval()
    # Model without weights
    dinov3 = AutoModel.from_config(AutoConfig.from_pretrained(model_id))
    backbone = DinoV3Backbone(dinov3)

    # Build model with same architecture as training
    model = ClockModel(
        backbone,
        # backbone_channels=backbone.blocks[-1].mlp.fc2.out_features,  # type:ignore[reportArgumentType]
        backbone_channels=backbone.hidden_size,
        d_model=1024,
        nhead=16,
        depth=3,
        dim_ff=None,
        dropout=0.0,
        polar_theta=64,
        polar_r=32,
        deform_points=8,
    ).to(device)

    # Load trained weights
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(image_tensor)

    # Extract predictions
    output = output.cpu().squeeze(0)  # Remove batch dimension
    sin_h, cos_h, sin_m, cos_m = output.tolist()

    # Convert to time
    hour, minute = sincos_to_time(sin_h, cos_h, sin_m, cos_m)

    # Print results
    print(f"Predicted Time: {hour:02d}:{minute:02d}")
    print("Raw predictions:")
    print(f"  Hour   -> sin: {sin_h:+.4f}, cos: {cos_h:+.4f}")
    print(f"  Minute -> sin: {sin_m:+.4f}, cos: {cos_m:+.4f}")


if __name__ == "__main__":
    main()
