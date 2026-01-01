import math
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.basic_transform import to_floats, to_sincos
from src.settings import p_env

plt.style.use("seaborn-v0_8")

# Set random seeds
SEED = p_env.RANDOM_SEED
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

has_cuda = str(DEVICE) == "cuda"

# TARGET_DIM = 224
TARGET_DIM = 448
DTYPE = torch.float32
NUM_EPOCHS = 64
# UNFREEZE_AT_EPOCH = 32
UNFREEZE_AT_EPOCH = 16
# BATCH_SIZE = 256
BATCH_SIZE = 512
# LEARNING_RATE = 1e-4
LEARNING_RATE = 6e-4
PRINT_EVERY_N_EPOCH = 1

print(f"Device: {DEVICE}, dtype: {DTYPE}")


class ClockDataset(Dataset):
    """Dataset for analog clock images."""

    def __init__(self, image_dir: Path, augment: bool = False, max_samples: int | None = None):
        self.image_paths = sorted(list(Path(image_dir).glob("*.png")))
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]

        # Build transforms
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.uint8, scale=False),
                    transforms.RandomResize(TARGET_DIM, int(TARGET_DIM * 1.2)),
                    transforms.RandomCrop((TARGET_DIM, TARGET_DIM)),
                    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToDtype(DTYPE, scale=True),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.uint8, scale=False),
                    transforms.Resize(TARGET_DIM),
                    transforms.CenterCrop((TARGET_DIM, TARGET_DIM)),
                    transforms.ToDtype(DTYPE, scale=True),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):  # type:ignore[invalid-method-override]
        img_path = self.image_paths[idx]

        # Load image
        image_pil = Image.open(img_path).convert("RGB")
        image = self.transform(image_pil)
        image_pil.close()

        # Parse time from filename: XXXXXX-HH_MM_SS.png
        time_str = img_path.stem.split("-")[1]  # Get HH_MM_SS part
        hour_float, minute_float = to_floats(time_str)

        # Convert to sin/cos
        sin_h, cos_h = to_sincos(hour_float)
        sin_m, cos_m = to_sincos(minute_float)

        # Target: [sin_hour, cos_hour, sin_minute, cos_minute]
        target = torch.tensor([sin_h, cos_h, sin_m, cos_m], dtype=DTYPE)

        return image, target


class DinoBilinear(nn.Module):
    """DINO-v2 with bilinear interpolation."""

    def __init__(self, dino):
        super().__init__()
        self.cls_token = dino.cls_token
        self.pos_embed = dino.pos_embed
        self.patch_size = dino.patch_size
        self.interpolate_offset = dino.interpolate_offset

        self.N = self.pos_embed.shape[1] - 1
        self.M = int(math.sqrt(self.N))

        self.patch_embed = dino.patch_embed
        self.blocks = nn.Sequential(*dino.blocks)
        self.norm = dino.norm
        self.head = dino.head

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        if npatch == self.N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        sx = float(w0 + self.interpolate_offset) / self.M
        sy = float(h0 + self.interpolate_offset) / self.M

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, self.M, self.M, dim).permute(0, 3, 1, 2),
            mode="bilinear",
            scale_factor=(sx, sy),
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, x):
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.blocks(x)
        x_norm = self.norm(x)
        x = x_norm[:, 0]
        x = self.head(x)
        return x


class ClockModel(nn.Module):
    """Clock reading model with frozen DINOv2 backbone."""

    def __init__(
        self,
        backbone,
        backbone_channels: int = 768,
        hidden_layers: list[int] = [256, 128],
    ):
        super().__init__()

        # Normalization for DINOv2
        self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))

        self.backbone = backbone

        # Simple MLP head
        self.head = nn.Sequential(
            nn.Linear(backbone_channels, hidden_layers[0]),
            nn.GELU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.GELU(),
            nn.Linear(hidden_layers[-1], 4),  # sin_h, cos_h, sin_m, cos_m
            nn.Tanh(),  # Bound outputs to [-1, 1] like sin/cos
        )

    def forward(self, x):
        # Normalize
        x = (x - self.norm_mean) / self.norm_std
        # Backbone
        x = self.backbone(x)
        # Head
        x = self.head(x)
        return x


def angle_error(pred_sin, pred_cos, true_sin, true_cos):
    """Compute angle error in radians between predicted and true angles."""
    pred_angle = torch.atan2(pred_sin, pred_cos)
    true_angle = torch.atan2(true_sin, true_cos)
    # Compute shortest angular distance
    diff = pred_angle - true_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.abs(diff)


@torch.inference_mode()
def validate(model, val_loader):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_hour_error = 0.0
    total_minute_error = 0.0
    n_batches = 0

    for images, targets in val_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Forward
        preds = model(images)

        # Loss
        loss = nn.functional.mse_loss(preds, targets)
        total_loss += loss.item()

        # Angle errors
        hour_error = angle_error(preds[:, 0], preds[:, 1], targets[:, 0], targets[:, 1])
        minute_error = angle_error(preds[:, 2], preds[:, 3], targets[:, 2], targets[:, 3])

        total_hour_error += hour_error.mean().item()
        total_minute_error += minute_error.mean().item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_hour_error = total_hour_error / n_batches
    avg_minute_error = total_minute_error / n_batches

    return avg_loss, avg_hour_error, avg_minute_error


def train_epoch(model, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Forward
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=has_cuda):
            preds = model(images)
            loss = nn.functional.mse_loss(preds, targets)

        # Backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    print("\n=== Loading datasets ===")
    # Use full datasets
    train_dataset = ClockDataset(p_env.TRAINING_DIR, augment=True, max_samples=None)
    val_dataset = ClockDataset(p_env.VALIDATION_DIR, augment=False, max_samples=None)

    num_workers = 8 if has_cuda else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    print("\n=== Loading DINOv2 backbone ===")
    # map_location="cpu"
    # Load pretrained DINOv2
    # 300M
    backbone_original = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    # 86M params
    # backbone_original = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    # 21M params
    # backbone_original = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # backbone_original = backbone_original.eval()

    backbone = DinoBilinear(backbone_original)

    # Build model
    model = ClockModel(
        backbone,
        backbone_channels=backbone.blocks[-1].mlp.fc2.out_features,  # type:ignore[reportArgumentType]
        hidden_layers = [512, 256],
    ).to(DEVICE)

    if has_cuda:
        model.compile()

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer
    warmup_steps = 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-7, end_factor=1.0, total_iters=warmup_steps
    )

    print("\n=== Training ===")
    start_time = perf_counter()

    # Track metrics for plotting
    history = []

    try:
        epoch = 0
        # Initial val loss
        val_loss, hour_err, minute_err = validate(model, val_loader)
        _scores = {
            "epoch": epoch,
            "train_loss": 0.5,
            "val_loss": val_loss,
            "hour_error_rad": hour_err,
            "minute_error_rad": minute_err,
        }
        history.append(_scores)
        print(" │ ".join(f"{k.replace('_', ' ').capitalize()} {v:>7.3f}" for k, v in _scores.items()))

        for epoch in range(NUM_EPOCHS):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)

            val_loss, hour_err, minute_err = validate(model, val_loader)
            _scores = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "hour_error_rad": hour_err,
                "minute_error_rad": minute_err,
            }
            history.append(_scores)

            # Print val scores every X epochs
            if epoch % PRINT_EVERY_N_EPOCH == 0:
                print(" │ ".join(f"{k.replace('_', ' ').capitalize()} {v:>7.3f}" for k, v in _scores.items()))

            # Unfreeze backbone
            if epoch == UNFREEZE_AT_EPOCH:
                print("Unfreezing backbone")
                for param in model.backbone.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-7, end_factor=1.0, total_iters=warmup_steps
                )

    except KeyboardInterrupt:
        if epoch == 0:
            return
        print(f"Stopping at {epoch=}")

    elapsed = perf_counter() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Final validation
    print("\n=== Final validation ===")
    val_loss, hour_err, minute_err = validate(model, val_loader)
    print(f"Final val loss: {val_loss:.4f}")
    print(f"Hour hand error: {hour_err:.4f} rad ({math.degrees(hour_err):.2f}°)")
    print(f"Minute hand error: {minute_err:.4f} rad ({math.degrees(minute_err):.2f}°)")

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)

    # Save model as safetensors
    output_path = results_dir / "checkpoint.safetensors"
    save_file(model.state_dict(), output_path)
    print(f"\nModel saved to {output_path}")

    # Plot training history
    if history:
        df = pd.DataFrame(history).set_index("epoch")

        _, axes = plt.subplots(dpi=220, figsize=(13, 3), ncols=2, sharex=True)

        # Loss plot
        df[["train_loss", "val_loss"]].plot(ax=axes[0], linewidth=1.4)
        axes[0].set_title("Losses")
        axes[0].set_xlabel(f"Epoch\n\nTotal wall clock: {elapsed:.1f}s")
        axes[0].set_ylabel("MSE Loss")
        axes[0].grid(True, alpha=0.3)

        # Error plot (convert to degrees)
        df_deg = df[["hour_error_rad", "minute_error_rad"]] * 180 / np.pi
        df_deg.plot(ax=axes[1], linewidth=1.4)
        axes[1].set_title("Angle Errors on Validation Set")
        axes[1].set_xlabel(f"Epoch\n\nTotal wall clock: {elapsed:.1f}s")
        axes[1].set_ylabel("Error (degrees)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = results_dir / "training_losses.png"
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Training plot saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()
