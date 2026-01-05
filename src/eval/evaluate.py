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
    checkpoint_path: str | Path,
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
        mlflow.start_run(run_id=run_id, nested=True)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Data
    val_ds = ClDataset(data_dir, split="val")
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
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
        checkpoint_path,
        scheduler=None,
        model=model,
        optimizer=None,
        scaler=None,
        map_location=device,
    )

    model.eval()

    # Storage
    all_clean_mean = []
    all_clean_logvar = []
    all_clean_true = []

    
    all_noise_mean = []    
    all_noise_logvar = []  
    all_noise_true = []    

    for x, (y_clean, y_noise) in val_loader:
        x = x.to(device)
        y_clean = y_clean.to(device)
        y_noise = y_noise.to(device)

        clean_mean, clean_logvar, noise_mean, noise_logvar = model(x)

        all_clean_mean.append(clean_mean.cpu())
        all_clean_true.append(y_clean.cpu())
        if clean_logvar is not None:
            all_clean_logvar.append(clean_logvar.cpu())

        all_noise_mean.append(noise_mean.cpu())
        all_noise_true.append(y_noise.cpu())
        if noise_logvar is not None:
            all_noise_logvar.append(noise_logvar.cpu())
        

    clean_mean_norm = torch.cat(all_clean_mean) # (N, L)
    clean_true_norm = torch.cat(all_clean_true)

    noise_mean_norm = torch.cat(all_noise_mean)
    noise_true_norm = torch.cat(all_noise_true)

    clean_mean_orig = val_ds.inverse_transform(clean_mean_norm)
    clean_true_orig = val_ds.inverse_transform(clean_true_norm)

    noise_mean_orig = val_ds.inverse_transform(noise_mean_norm)
    noise_true_orig = val_ds.inverse_transform(noise_true_norm)

    if predict_variance:
        clean_logvar_norm = torch.cat(all_clean_logvar)
        noise_logvar_norm = torch.cat(all_noise_logvar)

        clean_var_norm = torch.exp(clean_logvar_norm)
        noise_var_norm = torch.exp(noise_logvar_norm)

        clean_var_orig = val_ds.var_denormalize(clean_var_norm)
        noise_var_orig = val_ds.var_denormalize(noise_var_norm)
    else:
        clean_var_orig = None
        noise_var_orig = None

    # ----------- Metrics ----------
    clean_mse = torch.mean((clean_mean_orig - clean_true_orig) ** 2).item()
    noise_mse = torch.mean((noise_mean_orig - noise_true_orig) ** 2).item()

    print(f"Clean MSE (original units): {clean_mse:.6e}")
    print(f"Noise MSE (original units): {noise_mse:.6e}")

    # ---------- Plots ----------
    out_dir = Path("eval_outputs")
    out_dir.mkdir(exist_ok=True)

    ell = np.arange(seq_len)

    # Mean Prediction vs True (average over samples)
    # Clean
    plt.figure()
    plt.plot(ell, clean_true_orig.mean(dim=0), label="Clean True", marker="o", markersize=2)
    plt.plot(ell, clean_mean_orig.mean(dim=0), label="Clean Predicted", marker="o", markersize=2)
    plt.xlabel("l")
    plt.ylabel("Cl")
    plt.legend()
    plt.title("Clean Mean Cl")
    plt.tight_layout()
    plt.savefig(out_dir/ "clean_mean_vs_true.png")
    plt.close()

    # Noise
    plt.figure()
    plt.plot(ell, noise_true_orig.mean(dim=0), label="Noise True", marker="o", markersize=2)
    plt.plot(ell, noise_mean_orig.mean(dim=0), label="Noise Predicted", marker="o", markersize=2)
    plt.xlabel("l")
    plt.ylabel("Cl")
    plt.legend()
    plt.title("Noise Mean Cl")
    plt.tight_layout()
    plt.savefig(out_dir/ "noise_mean_vs_true.png")
    plt.close()

    # Sample realizations
    # Clean
    if clean_var_orig is not None:
        idx = 0
        eps = torch.randn_like(clean_mean_orig[idx])
        sampled = clean_mean_orig[idx] + torch.sqrt(clean_var_orig[idx])*eps

        plt.figure()
        plt.plot(ell, clean_true_orig[idx], label="True Clean", marker="o", markersize=2)
        plt.plot(ell, clean_mean_orig[idx], label="Pred clean mean", marker="o", markersize=2)
        plt.plot(ell, sampled, label="Sampled Clean", marker="o", markersize=2)
        plt.xlabel("l")
        plt.ylabel("Cl")
        plt.legend()
        plt.title("Sampled Clean realization")
        plt.tight_layout()
        plt.savefig(out_dir/ f"clean_sample_realization_{idx}.png")
        plt.close()

    # Noise
    if noise_var_orig is not None:
        idx = 0
        eps = torch.randn_like(noise_mean_orig[idx])
        sampled = noise_mean_orig[idx] + torch.sqrt(noise_var_orig[idx])*eps

        plt.figure()
        plt.plot(ell, noise_true_orig[idx], label="True Noise", marker="o", markersize=2)
        plt.plot(ell, noise_mean_orig[idx], label="Pred Noise mean", marker="o", markersize=2)
        plt.plot(ell, sampled, label="Sampled Noise", marker="o", markersize=2)
        plt.xlabel("l")
        plt.ylabel("Cl")
        plt.legend()
        plt.title("Sampled Noise realization")
        plt.tight_layout()
        plt.savefig(out_dir/ f"noise_sample_realization_{idx}.png")
        plt.close()

    if mlflow.active_run():
        mlflow.log_artifacts("eval_outputs", artifact_path="evaluation")
        mlflow.end_run()


if __name__ == "__main__":
    evaluate(
        data_dir="data/processed",
        checkpoint_path="experiments/cosmoformer_2layer/best_model.pt",
        seq_len=127,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=2,
        predict_variance=True,
    )