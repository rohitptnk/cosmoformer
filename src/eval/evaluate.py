from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.dataset import ClDataset
from src.models.transformer import Transformer1DAutoencoder
from src.utils.checkpoint import load_checkpoint

import mlflow


@torch.no_grad()
def evaluate(
    data_dir: str | Path,
    chechkpoint_path: str | Path,
    seq_len: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    batch_size: int = 128,
    predict_variance: bool = True,
    device_str: str = "cuda",
    run_id: str | None = None,
):
    if run_id is not None:
        mlflow.start_run(run_id=run_id)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Data
    val_ds = ClDataset(data_dir, split="val")
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Model
    model = Transformer1DAutoencoder(
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=0.0,
        predict_variance=predict_variance,
    ).to(device)

    load_checkpoint(
        chechkpoint_path,
        model=model,
        optimizer=None,
        scaler=None,
        map_location=device,
    )

    model.eval()

    # Storage
    all_mean = []
    all_logvar = []
    all_true = []

    for x,y in val_loader:
        x = x.to(device)
        y = y.to(device)

        mean, logvar = model(x)

        all_mean.append(mean.cpu())
        if logvar is not None:
            all_logvar.append(logvar.cpu())
        all_true.append(y.cpu())

    mean_norm = torch.cat(all_mean) # (N, L)
    true_norm = torch.cat(all_true)

    mean_orig = val_ds.inverse_transform(mean_norm)
    true_orig = val_ds.inverse_transform(true_norm)

    if predict_variance:
        logvar_norm = torch.cat(all_logvar)
        var_norm = torch.exp(logvar_norm)
        var_orig = val_ds.var_denormalize(var_norm)
    else:
        var_orig = None

    # ----------- Metrics ----------
    mse = torch.mean((mean_orig - true_orig) ** 2).item()
    print(f"MSE (original units): {mse:.6e}")

    # empirical residual variance per
    residual_var = torch.mean((mean_orig - true_orig) ** 2, dim=0)

    # ---------- Plots ----------
    out_dir = Path("eval_outputs")
    out_dir.mkdir(exist_ok=True)

    ell = np.arange(seq_len)

    # Mean Prediction vs True (average over samples)
    plt.figure()
    plt.plot(ell, true_orig.mean(dim=0), label="True")
    plt.plot(ell, mean_orig.mean(dim=0), label="Predicted")
    plt.xlabel("l")
    plt.ylabel("Cl")
    plt.legend
    plt.title("Mean Cl")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_dir/ "mean_vs_true.png")
    plt.close()

    # Residual variance vs l
    plt.figure()
    plt.plot(ell, residual_var.numpy(), label="Empirical residual var")
    if var_orig is not None:
        plt.plot(ell, var_orig.mean(dim=0).numpy(), label="Predicted var")
    plt.xlabel("l")
    plt.ylabel("Variance")
    plt.legend()
    plt.title("Variance comparision")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_dir/ "residual_variance_vs_l.png")
    plt.close()

    # Sample realizations
    if var_orig is not None:
        idx = 0
        eps = torch.randn_like(mean_orig[idx])
        sampled = mean_orig[idx] + torch.sqrt(var_orig[idx])*eps

        plt.figure()
        plt.plot(ell, true_orig[idx], label="True")
        plt.plot(ell, mean_orig[idx], label="Pred mean")
        plt.plot(ell, sampled, label="Sampled")
        plt.xlabel("l")
        plt.ylabel("Cl")
        plt.legend()
        plt.title("Sampled realization")
        plt.tight_layout()
        plt.show()
        plt.savefig(out_dir/ f"sample_realization_{idx}.png")
        plt.close()

    if mlflow.active_run():
        mlflow.log_artifacts("eval_outputs", artifact_path="evaluation")
        mlflow.end_run()


if __name__ == "__main__":
    evaluate(
        data_dir="data/processed",
        chechkpoint_path="checkpoints/epoch_020.pt",
        seq_len=31,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=2,
        predict_variance=True,
    )