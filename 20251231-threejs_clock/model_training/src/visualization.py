import math

import matplotlib.pyplot as plt
import torch


def sincos_to_time(sin_h, cos_h, sin_m, cos_m):
    """Convert sin/cos values to hours and minutes."""
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
def plot_predictions(model, dataset, device, output_path, num_samples=4):
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
        image_batch = image.unsqueeze(0).to(device)
        pred = model(image_batch).cpu().squeeze(0)

        # Convert to time
        pred_hours, pred_minutes = sincos_to_time(pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item())
        true_hours, true_minutes = sincos_to_time(
            target[0].item(), target[1].item(), target[2].item(), target[3].item()
        )

        # Plot image
        ax = axes[idx]
        # Convert from CHW to HWC and denormalize
        image_np = image.permute(1, 2, 0).numpy()
        ax.imshow(image_np)
        ax.axis("off")

        # Add title with prediction and ground truth
        title = f"True: {int(true_hours):02d}:{int(true_minutes):02d}\n"
        title += f"Pred: {int(pred_hours):02d}:{int(pred_minutes):02d}"
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Predictions plot saved to {output_path}")
