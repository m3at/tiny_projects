import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models import ClockModel


def sincos_to_time(elem: torch.Tensor):
    """Convert sin/cos values to hours and minutes."""
    sin_h, cos_h, sin_m, cos_m = (elem[0].item(), elem[1].item(), elem[2].item(), elem[3].item())

    # Get angles in radians
    hour_angle = math.atan2(sin_h, cos_h)
    minute_angle = math.atan2(sin_m, cos_m)

    # Convert to [0, 2π]
    if hour_angle < 0:
        hour_angle += 2 * math.pi
    if minute_angle < 0:
        minute_angle += 2 * math.pi

    # Convert to hours/minutes
    hours = (hour_angle / (2 * math.pi)) * 12
    minutes = (minute_angle / (2 * math.pi)) * 60

    return hours, minutes


@torch.inference_mode()
def plot_predictions(model: ClockModel, dataset, device, output_path, num_samples=4):
    """Plot predictions on sample images.

    Args:
        model: The model to use for predictions
        dataset: The dataset to sample from
        device: The device to run predictions on
        output_path: Path to save the plot
        num_samples: Number of samples to plot (default 4 for 2x2 grid)
    """
    model.eval()

    # Sample random indices
    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    # Create figure
    _, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
    axes = axes.flatten()

    for idx, sample_idx in enumerate(indices):
        image, target = dataset[sample_idx]

        # Get prediction
        pred = model.forward(image.unsqueeze(0).to(device)).cpu().squeeze(0)

        # Convert to time
        pred_h, pred_m = sincos_to_time(pred)
        true_h, true_m = sincos_to_time(target)

        # Plot image
        ax = axes[idx]
        # Convert from CHW to HWC and denormalize
        image_np = image.permute(1, 2, 0).numpy()
        ax.imshow(image_np)
        ax.axis("off")

        # Add title with prediction and ground truth
        title = f"True: {int(true_h):02d}:{int(true_m):02d}\n"
        title += f"Pred: {int(pred_h):02d}:{int(pred_m):02d}\n"
        title += f"Raw tensor: {pred.numpy().round(3)}"
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Predictions plot saved to {output_path}")


def plot_losses(df: pd.DataFrame, output_path: Path, elapsed: float, unfreeze_at_epoch: int):
    max_x = len(df)

    _, axes = plt.subplots(dpi=220, figsize=(10, 6), nrows=2, sharex=True)

    # Loss plot
    df[["train_loss", "val_loss"]].plot(ax=axes[0], linewidth=1.4)
    axes[0].set_title("Losses")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")

    # Error plot (convert to degrees)
    df_deg = df[["hour_error_rad", "minute_error_rad"]] * 180 / np.pi
    df_deg.plot(ax=axes[1], linewidth=1.4)
    axes[1].set_title("Angle Errors on Validation Set")
    axes[1].set_ylabel("Error (degrees)")
    axes[1].set_yscale("log")

    axes[1].set_xlabel(f"Epoch\n\nTotal wall clock: {elapsed:.1f}s")
    axes[1].set_xlim(0, max_x)

    if unfreeze_at_epoch < max_x:
        for ax in axes:
            ax.axvline(unfreeze_at_epoch, linestyle="--", alpha=0.7, linewidth=1.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Training plot saved to {output_path}")
    plt.close()
