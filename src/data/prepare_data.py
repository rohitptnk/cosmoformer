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
    mixed3_name = cfg["data"].get("mixed3_name", "mixed3.npy")
    mixed4_name = cfg["data"].get("mixed4_name", "mixed4.npy")
    true_name = cfg["data"].get("true_name", "true.npy")
    freq1_name = cfg["data"].get("freq1_name", "freq1.npy")
    freq2_name = cfg["data"].get("freq2_name", "freq2.npy")
    freq3_name = cfg["data"].get("freq3_name", "freq3.npy")
    freq4_name = cfg["data"].get("freq4_name", "freq4.npy")

    X1 = np.load(raw_dir / mixed1_name)
    X2 = np.load(raw_dir / mixed2_name)
    X3 = np.load(raw_dir / mixed3_name)
    X4 = np.load(raw_dir / mixed4_name)
    Y_true = np.load(raw_dir / true_name)
    Y_freq1 = np.load(raw_dir / freq1_name)
    Y_freq2 = np.load(raw_dir / freq2_name)
    Y_freq3 = np.load(raw_dir / freq3_name)
    Y_freq4 = np.load(raw_dir / freq4_name)

    print("raw mixed1:", X1.shape)
    print("raw mixed2:", X2.shape)
    print("raw mixed3:", X3.shape)
    print("raw mixed4:", X4.shape)
    print("raw true:", Y_true.shape)
    print("raw freq1:", Y_freq1.shape)
    print("raw freq2:", Y_freq2.shape)
    print("raw freq3:", Y_freq3.shape)
    print("raw freq4:", Y_freq4.shape)

    if not (X1.shape == X2.shape == X3.shape == X4.shape == Y_true.shape == Y_freq1.shape == Y_freq2.shape == Y_freq3.shape == Y_freq4.shape):
        raise ValueError(
            f"Inconsistent shapes: mixed1 {X1.shape}, mixed2 {X2.shape}, mixed3 {X3.shape}, mixed4 {X4.shape}, "
            f"true {Y_true.shape}, freq1 {Y_freq1.shape}, freq2 {Y_freq2.shape}, freq3 {Y_freq3.shape}, freq4 {Y_freq4.shape}"
        )

    # random split
    N = X1.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = int(train_frac * N)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    def process_and_save(arr, name, train_idx, val_idx, out_dir):
        # Compute mean and std exclusively from training split
        mean = float(arr[train_idx].mean())
        std = float(arr[train_idx].std())
        std = max(std, eps)

        arr_train = (arr[train_idx] - mean) / std
        arr_val = (arr[val_idx] - mean) / std
        
        np.save(out_dir / f"{name}_train.npy", arr_train)
        np.save(out_dir / f"{name}_val.npy", arr_val)
        
        return arr_train.shape, arr_val.shape, mean, std

    shapes = {}
    stats = {}
    
    for arr, name in [
        (X1, "X1"), (X2, "X2"), (X3, "X3"), (X4, "X4"), 
        (Y_true, "Y_true"), (Y_freq1, "Y_freq1"), (Y_freq2, "Y_freq2"), (Y_freq3, "Y_freq3"), (Y_freq4, "Y_freq4")
    ]:
        tr_shape, val_shape, mean, std = process_and_save(arr, name, train_idx, val_idx, out_dir)
        shapes[name] = (tr_shape, val_shape)
        stats[name] = {"mean": mean, "std": std}

    print("\n--- Processed Data ---")
    for key, (tr_shape, val_shape) in shapes.items():
        print(f"{key}: train {tr_shape}, val {val_shape}")

    print("\n--- Computed Scalers ---")
    for key, stat in stats.items():
        print(f"{key}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")
        np.save(out_dir / f"{key}_scaler_mean.npy", np.array([stat['mean']], dtype=np.float64))
        np.save(out_dir / f"{key}_scaler_std.npy", np.array([stat['std']], dtype=np.float64))

    print(f"\nSaved processed data to {out_dir}")
    return out_dir

if __name__ == "__main__":

    config_path = "configs/config.yaml"
    prepare_and_save(config_path)

    
