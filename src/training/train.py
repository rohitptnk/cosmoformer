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

from src.utils.checkpoint import save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import mlflow
from src.utils.mlflow_utils import setup_mlflow
from src.utils.scheduler import build_scheduler
from src.utils.losses import build_loss


# Training / validation loops
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    use_amp: bool,
    loss_fn,
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
                loss = (loss_fn(mean, logvar, y))
            
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            mean, logvar = model(x)
            loss = (loss_fn(mean, logvar, y))
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)    


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn,
) -> float:
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        mean, logvar = model(x)
        loss = (loss_fn(mean, logvar, y))
        total_loss += loss.item()

    return total_loss / len(loader)


# Main training entry
def train(cfg: dict):

    # Load Config
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]
    optim_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    ckpt_cfg = cfg["checkpoint"]

    seed = exp_cfg["seed"]
    device = exp_cfg["device"]

    data_dir = data_cfg["processed_dir"]
    seq_len = data_cfg["seq_len"]
    pin_memory = data_cfg["pin_memory"]
    num_workers = data_cfg["num_workers"]
    batch_size = data_cfg["batch_size"]

    d_model = model_cfg['d_model']
    n_heads = model_cfg['n_heads']
    d_ff = model_cfg['d_ff']
    n_layers = model_cfg['n_layers']
    dropout = model_cfg['dropout']
    predict_variance = model_cfg['predict_variance']

    use_amp = train_cfg["use_amp"] and device.type == "cuda"
    epochs = train_cfg["epochs"]

    optim_name = optim_cfg["name"]
    lr = optim_cfg["lr"]
    weight_decay = optim_cfg["weight_decay"]

    use_mlflow = log_cfg["use_mlflow"]
    experiment_name = log_cfg["experiment_name"]
    run_name = log_cfg["run_name"]    

    resume_from = ckpt_cfg["resume_from"]
    checkpoint_dir = ckpt_cfg["checkpoint_dir"]
    loss_fn = build_loss(cfg)


    # Set Values
    torch.manual_seed(seed=seed)
    device = torch.device(device=device)
    print(f"Using device: {device}")

    mlflow_run = None
    if use_mlflow:
        mlflow_run = setup_mlflow(
            experiment_name=experiment_name,
            run_name=run_name,
        )

        run_id = mlflow_run.info.run_id
        print(f"MLflow run_id: {run_id}")
        mlflow.log_artifact(config_path, artifact_path="config")


    # Datasets
    train_ds = ClDataset(data_dir, split="train")
    val_ds = ClDataset(data_dir, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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

    # ---------- OPTIMIZER ----------
    OPTIMIZERS = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }
    if optim_name not in OPTIMIZERS:
        raise ValueError(f"Unsupported Optimizer: {optim_name}")
    
    optim_cls = OPTIMIZERS[optim_name]
    optimizer = optim_cls(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scaler = GradScaler() if use_amp else None

    # ---------- Scheduler ----------
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_cfg=sched_cfg,
        total_steps=total_steps
    )

    # Resume
    start_epoch=1
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        start_epoch = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        ) + 1

    # Training Loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, epochs+1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            loss_fn,
        )
        val_loss = validate(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} |"
            f"Val Loss: {val_loss:.6f}"
        )

        if use_mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model found (val_loss={val_loss:.6f}). Saving checkpoint.")

            save_checkpoint(
                path=f"{checkpoint_dir}/best_model.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
            )
        
            if use_mlflow:
                mlflow.log_artifact(
                    f"{checkpoint_dir}/best_model.pt",
                    artifact_path="checkpoints",
                )
                
    if use_mlflow:
        mlflow.end_run()

    return model


# Script usage
if __name__ == "__main__":
    from utils.config_utils import load_config

    config_path = "configs/config_2layer.yaml"
    cfg = load_config(config_path)
    train(cfg)