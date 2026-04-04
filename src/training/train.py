from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from src.data.dataset import ClDataset
from src.models.transformer import Transformer1DAutoencoder

from src.utils.checkpoint import save_checkpoint, load_checkpoint

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

    for (x1, x2, x3, x4), (y_true, y_freq1, y_freq2, y_freq3, y_freq4) in loader:
        x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)
        y_true = y_true.to(device)
        y_freq1 = y_freq1.to(device)
        y_freq2 = y_freq2.to(device)
        y_freq3 = y_freq3.to(device)
        y_freq4 = y_freq4.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type=device.type):
                outputs = model(x1, x2, x3, x4)
                loss = loss_fn(*outputs, y_true, y_freq1, y_freq2, y_freq3, y_freq4)
            
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x1, x2, x3, x4)
            loss = loss_fn(*outputs, y_true, y_freq1, y_freq2, y_freq3, y_freq4)
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

    for (x1, x2, x3, x4), (y_true, y_freq1, y_freq2, y_freq3, y_freq4) in loader:
        x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)
        y_true = y_true.to(device)
        y_freq1 = y_freq1.to(device)
        y_freq2 = y_freq2.to(device)
        y_freq3 = y_freq3.to(device)
        y_freq4 = y_freq4.to(device)

        outputs = model(x1, x2, x3, x4)
        loss = loss_fn(*outputs, y_true, y_freq1, y_freq2, y_freq3, y_freq4)
        total_loss += loss.item()

    return total_loss / len(loader)


# Main training entry
def train(config_path: Union[str, Path]):

    cfg = load_config(config_path)

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
    device_str = exp_cfg["device"]
    device = torch.device(device=device_str)
    print(f"Using device: {device}")

    data_dir = data_cfg["processed_dir"]
    seq_len = data_cfg["seq_len"]
    pin_memory = data_cfg["pin_memory"]
    num_workers = data_cfg["num_workers"]

    d_model = model_cfg['d_model']
    n_heads = model_cfg['n_heads']
    d_ff = model_cfg['d_ff']
    n_layers = model_cfg['n_layers']
    dropout = model_cfg['dropout']
    predict_variance = model_cfg['predict_variance']

    use_amp = train_cfg["use_amp"] and device.type == "cuda"
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]

    optim_name = optim_cfg["name"]
    lr = optim_cfg["lr"]
    weight_decay = optim_cfg["weight_decay"]

    use_mlflow = log_cfg["use_mlflow"]
    experiment_name = log_cfg["mlflow"]["experiment_name"]
    run_name = log_cfg["mlflow"]["run_name"]    

    resume_from = ckpt_cfg["resume_from"]
    checkpoint_dir = ckpt_cfg["checkpoint_dir"]
    loss_fn = build_loss(cfg)


    # Set Values
    torch.manual_seed(seed=seed)
    

    mlflow_run = None
    if use_mlflow:
        mlflow_run = setup_mlflow(
            experiment_name=experiment_name,
            run_name=run_name,
        )

        run_id = mlflow_run.info.run_id
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(checkpoint_dir) / "run_id.txt", "w") as f:
            f.write(run_id)
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
    best_val_loss = float("inf")
    start_epoch=1
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        last_epoch, best_val_loss = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
            scheduler=scheduler,
        )
        start_epoch = last_epoch + 1

    # Training Loop
    
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
            f"Train Loss: {train_loss:.6f} | "
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
                best_val_loss=best_val_loss,
                scheduler=scheduler,
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
    from src.utils.config_utils import load_config
    import time

    start_time = time.time()

    config_path = "configs/config.yaml"
    print(f"Using config: {config_path}")
    print("Starting training...")
    try:
        train(config_path)
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training failed with exception: {e}")
        raise e
    
    end_time = time.time()
    elapsed_time = (end_time-start_time)/3600
    print(f"It took {elapsed_time:.2f} hours.")