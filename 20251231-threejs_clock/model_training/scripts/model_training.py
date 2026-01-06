import math
import random
from pathlib import Path
from time import perf_counter

import huggingface_hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import AutoModel

from src.dataset import ClockDataset
from src.models import ClockModel, DinoV3Backbone
from src.settings import p_env
from src.training import train_epoch, validate
from src.visualization import plot_losses, plot_predictions

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

# TARGET_DIM = 256
TARGET_DIM = 512
if has_cuda:
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32

NUM_EPOCHS = 500
DROP_LR_AT_EPOCHS = [300, 400]
LR_DROP_FACTOR = 3

BATCH_SIZE = 512
# LEARNING_RATE = 1e-4
LEARNING_RATE = 3e-4
PRINT_EVERY_N_EPOCH = 1

print(f"Device: {DEVICE}, dtype: {DTYPE}")


def main():
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)

    print(f"=== {TARGET_DIM=} {LEARNING_RATE=} ===")

    print("\n=== Loading datasets ===")
    # Use full datasets
    train_dataset = ClockDataset(p_env.TRAINING_DIR, augment=True, max_samples=None, target_dim=TARGET_DIM, dtype=DTYPE)
    val_dataset = ClockDataset(
        p_env.VALIDATION_DIR, augment=False, max_samples=None, target_dim=TARGET_DIM, dtype=DTYPE
    )

    num_workers = 12 if has_cuda else 0
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=has_cuda
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=has_cuda
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # DINO v2
    # # Load pretrained DINOv2
    # backbone_original = torch.hub.load(
    #     "facebookresearch/dinov2",
    #     # 300M
    #     # "dinov2_vitl14",
    #     # 86M params
    #     # "dinov2_vitb14",
    #     "dinov2_vitb14_reg",
    #     # 21M params
    #     # "dinov2_vits14",
    # )
    #
    # # backbone = DinoBilinear(backbone_original)
    # backbone = DinoBackbone(backbone_original)
    # backbone_channels = backbone.blocks[-1].mlp.fc2.out_features,  # type:ignore[reportArgumentType]

    # DINO v3
    # _k = "vitb16"
    _k = "vitl16"
    # _k = "vith16plus"
    _nb_params = {
        "vits16": "21M",
        "vits16plus": "29M",
        "vitb16": "86M",
        "vitl16": "300M",
        "vith16plus": "840M",
    }[_k]
    model_id = f"facebook/dinov3-{_k}-pretrain-lvd1689m"
    print(f"\n=== Preparing with DINOv3 {_k} ({_nb_params}) backbone ===")

    huggingface_hub.login(p_env.HF_TOKEN.get_secret_value())
    dinov3 = AutoModel.from_pretrained(model_id, dtype=DTYPE).eval()
    backbone = DinoV3Backbone(dinov3)
    backbone_channels = backbone.hidden_size

    # Build model
    model = (
        ClockModel(
            backbone,
            backbone_channels=backbone_channels,
            d_model=1024,
            nhead=16,
            depth=3,
            dim_ff=None,
            dropout=0.0,
            polar_theta=64,
            polar_r=32,
            deform_points=8,
        )
        .to(DTYPE)
        .to(DEVICE)
    )

    # Load pre-trained model if it exists
    if (baseline_path := results_dir / "baseline.safetensors").exists():
        print(f"Loading weights from {baseline_path}")
        state_dict = load_file(baseline_path)
        model.load_state_dict(state_dict)

    if has_cuda:
        print("Compiling...", end=" ")
        model.compile()
        # Run inference once
        _ = model(torch.rand((1, 3, TARGET_DIM, TARGET_DIM), dtype=DTYPE, device=DEVICE))
        print("Done")

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer
    warmup_steps = 32  # =2 epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, fused=has_cuda)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-7, end_factor=1.0, total_iters=warmup_steps
    )

    # Plot predictions before training
    # print("\n=== Plotting predictions before training ===")
    # plot_predictions(model, val_dataset, DEVICE, results_dir / "predictions_before.png")

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
        print(
            "      " + " │ ".join(f"{k.split('_', maxsplit=1)[0].capitalize()} {v:>7.3f}" for k, v in _scores.items())
        )
        _lapse = perf_counter()

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
                _delta = perf_counter() - _lapse
                print(
                    f"[{_delta:>3.0f}s] "
                    + " │ ".join(f"{k.split('_', maxsplit=1)[0].capitalize()} {v:>7.3f}" for k, v in _scores.items())
                )
                _lapse = perf_counter()

            # One-off learning rate drop instead of proper scheduling, good enough
            if epoch in DROP_LR_AT_EPOCHS:
                for group in optimizer.param_groups:
                    group["lr"] /= LR_DROP_FACTOR
                    # _lr = group["lr"]
                    # break
                # optimizer = torch.optim.AdamW(
                #     model.parameters(), lr=_lr / LR_DROP_FACTOR, weight_decay=0.0, fused=has_cuda
                # )
                # scheduler = torch.optim.lr_scheduler.LinearLR(
                #     optimizer, start_factor=1e-7, end_factor=1.0, total_iters=warmup_steps
                # )

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
        plot_losses(df, results_dir / "training_losses.png", elapsed, 9999, DROP_LR_AT_EPOCHS)


if __name__ == "__main__":
    main()
