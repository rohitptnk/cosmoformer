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
def evaluate(config_path: str):

    cfg = load_config(config_path)

    # Load Config
    data_dir = cfg["data"]["processed_dir"]
    seq_len = cfg["data"]["seq_len"]

    checkpoint_dir = cfg["checkpoint"]["checkpoint_dir"]
    checkpoint_path = Path(checkpoint_dir) / "best_model.pt"

    d_model = cfg["model"]["d_model"]
    n_heads = cfg["model"]["n_heads"]
    d_ff = cfg["model"]["d_ff"]
    n_layers = cfg["model"]["n_layers"]
    predict_variance = cfg["model"]["predict_variance"]

    batch_size = cfg["training"]["batch_size"]

    device_str = cfg["experiment"]["device"]

    use_mlflow = cfg["logging"].get("use_mlflow", False)

    if use_mlflow:
        run_id_path = Path(checkpoint_dir) / "run_id.txt"
        run_id = run_id_path.read_text().strip()
        print(run_id)
        mlflow.start_run(run_id=run_id)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    out_dir = Path(checkpoint_dir) / "eval_outputs"
    out_dir.mkdir(exist_ok=True)

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
    storage = {
        "clean": {"mean": [], "logvar": [], "true": []},
        "fg1": {"mean": [], "logvar": [], "true": []},
        "fg2": {"mean": [], "logvar": [], "true": []},
    }

    for (x1, x2), (y_clean, y_fg1, y_fg2) in val_loader:
        x1, x2 = x1.to(device), x2.to(device)
        y_clean = y_clean.to(device)
        y_fg1 = y_fg1.to(device)
        y_fg2 = y_fg2.to(device)

        c_m, c_v, f1_m, f1_v, f2_m, f2_v = model(x1, x2)

        storage["clean"]["mean"].append(c_m.cpu())
        storage["clean"]["true"].append(y_clean.cpu())
        if c_v is not None: storage["clean"]["logvar"].append(c_v.cpu())

        storage["fg1"]["mean"].append(f1_m.cpu())
        storage["fg1"]["true"].append(y_fg1.cpu())
        if f1_v is not None: storage["fg1"]["logvar"].append(f1_v.cpu())

        storage["fg2"]["mean"].append(f2_m.cpu())
        storage["fg2"]["true"].append(y_fg2.cpu())
        if f2_v is not None: storage["fg2"]["logvar"].append(f2_v.cpu())

    results = {}
    for key in storage:
        mean_norm = torch.cat(storage[key]["mean"])
        true_norm = torch.cat(storage[key]["true"])
        
        mean_orig = val_ds.inverse_transform(mean_norm)
        true_orig = val_ds.inverse_transform(true_norm)
        
        mse = torch.mean((mean_orig - true_orig) ** 2).item()
        print(f"{key.capitalize()} MSE (original units): {mse:.6e}")
        
        var_orig = None
        if predict_variance and storage[key]["logvar"]:
            logvar_norm = torch.cat(storage[key]["logvar"])
            var_norm = torch.exp(logvar_norm)
            var_orig = val_ds.var_denormalize(var_norm)
            
        results[key] = {
            "mean_orig": mean_orig,
            "true_orig": true_orig,
            "var_orig": var_orig,
            "mse": mse
        }

    # ---------- Plots ----------
    ell = np.arange(seq_len)
    eps = torch.randn_like(results["clean"]["mean_orig"])

    for key in results:
        res = results[key]
        t_orig = res["true_orig"]
        m_orig = res["mean_orig"]
        v_orig = res["var_orig"]

        # Mean Prediction vs True
        plt.figure()
        plt.errorbar(ell, t_orig.mean(dim=0), yerr=t_orig.std(dim=0), 
                     label=f"{key.capitalize()} True", fmt="-o", markersize=3, capsize=3)
        plt.errorbar(ell, m_orig.mean(dim=0), yerr=m_orig.std(dim=0), 
                     label=f"{key.capitalize()} Predicted", fmt="--o", markersize=3, capsize=3)
        plt.xlabel("l")
        plt.ylabel("Cl")
        plt.legend()
        plt.title(f"{key.capitalize()} Mean Cl")
        plt.tight_layout()
        plt.savefig(out_dir/ f"{key}_mean_vs_true.png")
        plt.close()

        # Sampled realizations
        if v_orig is not None:
            idx = 0
            sampled = m_orig[idx] + torch.sqrt(v_orig[idx]) * torch.randn_like(m_orig[idx])
            plt.figure()
            plt.plot(ell, t_orig[idx], label=f"True {key.capitalize()}")
            plt.plot(ell, sampled, label=f"Sampled {key.capitalize()}", linestyle="--")
            plt.xlabel("l")
            plt.ylabel("Cl")
            plt.legend()
            plt.title(f"Sampled {key.capitalize()} realization")
            plt.tight_layout()
            plt.savefig(out_dir/ f"{key}_sample_realization_{idx}.png")
            plt.close()

    if use_mlflow and mlflow.active_run() is not None:
        for key in results:
            mlflow.log_metric(f"{key}_mse", results[key]["mse"])
        mlflow.log_artifacts(str(out_dir), artifact_path="evaluation")
        mlflow.end_run()


if __name__ == "__main__":
    from src.utils.config_utils import load_config

    config_path = "configs/config_2layer.yaml"
    print(f"Using config: {config_path}")
    print("Starting evaluation...")

    evaluate(config_path)
    print("Evaluation successful.")