from pathlib import Path
import numpy as np
from src.utils.config_utils import load_config

def prepare_and_save(config_path, train_frac=0.8, seed=42, eps=1e-10):
    cfg = load_config(config_path)
    raw_dir= Path(cfg["data"]["raw_dir"])
    out_dir= Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed1_name = cfg["data"].get("mixed1_name", "mixed1.npy")
    mixed2_name = cfg["data"].get("mixed2_name", "mixed2.npy")
    true_name = cfg["data"].get("true_name", "true.npy")
    fg1_name = cfg["data"].get("fg1_name", "fg1.npy")
    fg2_name = cfg["data"].get("fg2_name", "fg2.npy")

    X1 = np.load(raw_dir / mixed1_name)
    X2 = np.load(raw_dir / mixed2_name)
    Y_true = np.load(raw_dir / true_name)
    Y_fg1 = np.load(raw_dir / fg1_name)
    Y_fg2 = np.load(raw_dir / fg2_name)

    print("raw mixed1:", X1.shape)
    print("raw mixed2:", X2.shape)
    print("raw true:", Y_true.shape)
    print("raw fg1:", Y_fg1.shape)
    print("raw fg2:", Y_fg2.shape)

    if not (X1.shape == X2.shape == Y_true.shape == Y_fg1.shape == Y_fg2.shape):
        raise ValueError(
            f"Inconsistent shapes: mixed1 {X1.shape}, mixed2 {X2.shape}, "
            f"true {Y_true.shape}, fg1 {Y_fg1.shape}, fg2 {Y_fg2.shape}"
        )

    # random split
    N = X1.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = int(train_frac * N)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # compute mean and std from mixed training data
    # We combine X1 and X2 training data to get a global mean/std for all signals
    X_train_combined = np.concatenate([X1[train_idx], X2[train_idx]], axis=0)
    mean = float(X_train_combined.mean())
    std = float(X_train_combined.std())
    std = max(std, eps)

    print(f"\nComputed Scaler: mean={mean:.4f}, std={std:.4f}")

    def process_and_save(arr, name, train_idx, val_idx, mean, std, out_dir):
        arr_train = (arr[train_idx] - mean) / std
        arr_val = (arr[val_idx] - mean) / std
        np.save(out_dir / f"{name}_train.npy", arr_train)
        np.save(out_dir / f"{name}_val.npy", arr_val)
        return arr_train.shape, arr_val.shape

    shapes = {}
    shapes['X1'] = process_and_save(X1, "X1", train_idx, val_idx, mean, std, out_dir)
    shapes['X2'] = process_and_save(X2, "X2", train_idx, val_idx, mean, std, out_dir)
    shapes['Y_true'] = process_and_save(Y_true, "Y_true", train_idx, val_idx, mean, std, out_dir)
    shapes['Y_fg1'] = process_and_save(Y_fg1, "Y_fg1", train_idx, val_idx, mean, std, out_dir)
    shapes['Y_fg2'] = process_and_save(Y_fg2, "Y_fg2", train_idx, val_idx, mean, std, out_dir)

    print("\n--- Processed Data ---")
    for key, (tr_shape, val_shape) in shapes.items():
        print(f"{key}: train {tr_shape}, val {val_shape}")

    np.save(out_dir / "scaler_mean.npy", np.array([mean], dtype=np.float64))   
    np.save(out_dir / "scaler_std.npy", np.array([std], dtype=np.float64))   

    print(f"\nSaved processed data to {out_dir}")
    return out_dir

if __name__ == "__main__":

    config_path = "configs/config_2layer.yaml"
    prepare_and_save(config_path)

    
