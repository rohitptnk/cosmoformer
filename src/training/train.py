from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from src.data.dataset import ClDataset
from src.models.transformer import Transformer1DAutoencoder


# Loss Functions
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred-target)**2)

def heteroscedastic_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Kendall & Gal heteroscedastic regression loss:
        exp(-s) * (y - μ)^2 + s
    where s = log variance
    """
    return torch.mean(torch.exp(-logvar) * (target-mean) ** 2 + logvar)   


# Training / validation loops
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                mean, logvar = model(x)
                loss = (
                    heteroscedastic_loss(mean, logvar, y)
                    if logvar is not None
                    else mse_loss(mean, y)
                )
            
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            mean, logvar = model(x)
            loss = (
                heteroscedastic_loss(mean, logvar, y)
                if logvar is not None
                else mse_loss(mean, y)
            )
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)    


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        mean, logvar = model(x)
        loss = (
            heteroscedastic_loss(mean, logvar, y)
            if logvar is not None
            else mse_loss(mean, y)
        )
        total_loss += loss.item()

    return total_loss / len(loader)


# Main training entry
def train(
    data_dir: str | Path,
    seq_len: int,
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
    n_layers: int = 2,
    batch_size: int = 64,
    lr: float = 1e-4,
    epochs: int = 20,
    dropout: float = 0.1,
    predict_variance: bool = True,
    use_amp: bool = True,
    num_workers: int = 4,
    seed: int = 42,
):
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp = use_amp and device.type == "cuda"

    # Datasets
    train_ds = ClDataset(data_dir, split="train")
    val_ds = ClDataset(data_dir, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


    # Model
    model = Transformer1DAutoencoder(
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
        predict_variance=predict_variance,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scaler = GradScaler() if use_amp else None


    # Training Loop
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
        )
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} |"
            f"Val Loss: {val_loss:.6f}"
        )

    return model


# Script usage
if __name__ == "__main__":
    train(
        data_dir="data/processed",
        seq_len=32,
        n_layers=2,
        predict_variance=True,
        use_amp=True,
        epochs=10,
    )