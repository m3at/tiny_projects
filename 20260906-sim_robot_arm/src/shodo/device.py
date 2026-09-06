"""Explicit, portable device selection for neural controllers; physics stays on CPU."""

import torch


def resolve_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("Device must be auto, cpu, cuda or mps")
    if requested == "cpu":
        return torch.device("cpu")
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested == "auto":
        return torch.device(next(name for name in ("cuda", "mps", "cpu") if available[name]))
    if not available[requested]:
        raise ValueError(f"Requested {requested} device is unavailable; use auto or cpu")
    return torch.device(requested)
