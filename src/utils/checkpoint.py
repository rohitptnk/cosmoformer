from pathlib import Path
from typing import Optional
from torch.amp.grad_scaler import GradScaler

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scaler: Optional[GradScaler] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    return ckpt.get("epoch", 0)