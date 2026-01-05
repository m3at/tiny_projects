import torch
from torch import nn


def angle_error(pred_sin, pred_cos, true_sin, true_cos):
    """Compute angle error in radians between predicted and true angles."""
    pred_angle = torch.atan2(pred_sin, pred_cos)
    true_angle = torch.atan2(true_sin, true_cos)
    # Compute shortest angular distance
    diff = pred_angle - true_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.abs(diff)


def angle_loss(pred_vec: torch.Tensor, true_vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Angle error in radians in [0, pi]."""
    pred_vec = nn.functional.normalize(pred_vec, dim=-1)
    true_vec = nn.functional.normalize(true_vec, dim=-1)

    # dot = cos(Δ)
    dot = (pred_vec * true_vec).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    # cross_z = sin(Δ) for 2D vectors
    cross = pred_vec[..., 0] * true_vec[..., 1] - pred_vec[..., 1] * true_vec[..., 0]
    return torch.atan2(cross.abs(), dot)


@torch.inference_mode()
def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_hour_error = 0.0
    total_minute_error = 0.0
    n_batches = 0

    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)

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


def train_epoch(model, train_loader, optimizer, scheduler, device, has_cuda):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        # Forward
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=has_cuda):
            # Simple MSE loss
            # preds = model(images)
            # loss = nn.functional.mse_loss(preds, targets)

            # Angle loss
            preds = model(images).view(-1, 2, 2)
            t = targets.view(-1, 2, 2)

            hour_ang = angle_loss(preds[:, 0], t[:, 0])
            min_ang = angle_loss(preds[:, 1], t[:, 1])

            # Concave power to amplify tiny angles
            min_ang = (min_ang + 1e-6).pow(0.5).mean()

            loss = 0.5 * hour_ang.mean() + 1.0 * min_ang.mean()

        # Backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches
