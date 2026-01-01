import math
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from src.dataset import ClockDataset
from src.models import ClockModel, DinoBilinear
from src.settings import p_env
from src.training import train_epoch, validate
from src.visualization import plot_predictions

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

TARGET_DIM = 224
# TARGET_DIM = 448
DTYPE = torch.float32
# NUM_EPOCHS = 64
NUM_EPOCHS = 300
# UNFREEZE_AT_EPOCH = 32
UNFREEZE_AT_EPOCH = 200
# UNFREEZE_AT_EPOCH = 1_000
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
# LEARNING_RATE = 6e-4
# LEARNING_RATE = 3e-3
PRINT_EVERY_N_EPOCH = 1

print(f"Device: {DEVICE}, dtype: {DTYPE}")


def main():
    print("\n=== Loading datasets ===")
    # Use full datasets
    train_dataset = ClockDataset(p_env.TRAINING_DIR, augment=True, max_samples=None, target_dim=TARGET_DIM, dtype=DTYPE)
    val_dataset = ClockDataset(
        p_env.VALIDATION_DIR, augment=False, max_samples=None, target_dim=TARGET_DIM, dtype=DTYPE
    )

    num_workers = 8 if has_cuda else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    print("\n=== Loading DINOv2 backbone ===")
    # map_location="cpu"
    # Load pretrained DINOv2
    # 300M
    # backbone_original = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    # 86M params
    backbone_original = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    # 21M params
    # backbone_original = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # backbone_original = backbone_original.eval()

    backbone = DinoBilinear(backbone_original)

    # Build model
    model = ClockModel(
        backbone,
        backbone_channels=backbone.blocks[-1].mlp.fc2.out_features,  # type:ignore[reportArgumentType]
        # d_model=512,
        # nhead=8,
        d_model=256,
        nhead=8,
        depth=2,
        dim_ff=None,
        dropout=0.0,
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

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)

    # Plot predictions before training
    print("\n=== Plotting predictions before training ===")
    plot_predictions(model, val_dataset, DEVICE, results_dir / "predictions_before.png")

    print("\n=== Training ===")
    start_time = perf_counter()

    # Track metrics for plotting
    history = []

    try:
        epoch = 0
        # Initial val loss
        val_loss, hour_err, minute_err = validate(model, val_loader, DEVICE)
        _scores = {
            "epoch": epoch,
            "train_loss": 1.0,
            "val_loss": val_loss,
            "hour_error_rad": hour_err,
            "minute_error_rad": minute_err,
        }
        history.append(_scores)
        print(" │ ".join(f"{k.replace('_', ' ').capitalize()} {v:>7.3f}" for k, v in _scores.items()))

        for epoch in range(NUM_EPOCHS):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, has_cuda)

            val_loss, hour_err, minute_err = validate(model, val_loader, DEVICE)
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
    val_loss, hour_err, minute_err = validate(model, val_loader, DEVICE)
    print(f"Final val loss: {val_loss:.4f}")
    print(f"Hour hand error: {hour_err:.4f} rad ({math.degrees(hour_err):.2f}°)")
    print(f"Minute hand error: {minute_err:.4f} rad ({math.degrees(minute_err):.2f}°)")

    # Plot predictions after training
    print("\n=== Plotting predictions after training ===")
    plot_predictions(model, val_dataset, DEVICE, results_dir / "predictions_after.png")

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
