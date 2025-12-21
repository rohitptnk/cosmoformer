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
    best_val_loss: float,
    scheduler,
    scaler: Optional[GradScaler] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }

    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    scheduler,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    map_location: str | torch.device = "cpu",
):
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    epoch = ckpt["epoch"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    return epoch, best_val_loss