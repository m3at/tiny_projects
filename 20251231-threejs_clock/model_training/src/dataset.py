from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset

from src.basic_transform import to_floats, to_sincos


class ClockDataset(Dataset):
    """Dataset for analog clock images."""

    def __init__(
        self,
        image_dir: Path,
        augment: bool = False,
        max_samples: int | None = None,
        target_dim: int = 448,
        dtype: torch.dtype = torch.float32,
    ):
        self.image_paths = sorted(list(Path(image_dir).glob("*.png")))
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]

        self.dtype = dtype

        # Build transforms
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.uint8, scale=False),
                    transforms.RandomResize(target_dim, int(target_dim * 1.2)),
                    transforms.RandomCrop((target_dim, target_dim)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                    transforms.ToDtype(dtype, scale=True),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.uint8, scale=False),
                    transforms.Resize(target_dim),
                    transforms.CenterCrop((target_dim, target_dim)),
                    transforms.ToDtype(dtype, scale=True),
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
        target = torch.tensor([sin_h, cos_h, sin_m, cos_m], dtype=self.dtype)

        return image, target
