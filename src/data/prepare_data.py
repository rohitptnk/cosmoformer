from pathlib import Path
import numpy as np
from src.utils.config_utils import load_config

def prepare_and_save(config_path, train_frac=0.8, seed=42, eps=1e-10):
    cfg = load_config(config_path)
    raw_dir= Path(cfg["data"]["raw_dir"])
    out_dir= Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed_cls_name = cfg["data"]["mixed_cls_name"]
    true_cls_name = cfg["data"]["true_cls_name"]
    noise_cls_name = cfg["data"]["noise_cls_name"]

    X = np.load(raw_dir / mixed_cls_name) # mixed = true + noise
    Y_true = np.load(raw_dir / true_cls_name)
    Y_noise = np.load(raw_dir / noise_cls_name)

    print("raw mixed:", X.shape)
    print("raw true:", Y_true.shape)
    print("raw noise:", Y_noise.shape)
    print()
    print("raw mixed mean:", X.mean())
    print("raw true mean:", Y_true.mean())
    print("raw noise mean:", Y_noise.mean())
    print()
    print("raw mixed std:", X.std())
    print("raw true std:", Y_true.std())
    print("raw noise std:", Y_noise.std())

    if not (X.shape == Y_true.shape == Y_noise.shape):
        raise ValueError(
            f"Inconsistent shapes: mixed {X.shape}, "
            f"true {Y_true.shape}, noise {Y_noise.shape}"
        )

    # random 80-20 train-test split
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = int(train_frac * N)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train = X[train_idx]
    Yc_train = Y_true[train_idx]
    Yn_train = Y_noise[train_idx]

    X_val = X[val_idx]
    Yc_val = Y_true[val_idx]
    Yn_val = Y_noise[val_idx]

    # compute mean and std
    mean = float(X_train.mean())
    std = float(X_train.std())
    std = max(std, eps)

    # standardize all splits
    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std

    Yc_train_s = (Yc_train - mean) / std
    Yc_val_s = (Yc_val - mean) / std

    Yn_train_s = (Yn_train - mean) / std
    Yn_val_s = (Yn_val - mean) / std

    print("mean:", mean)
    print("std:", std)

    print("\n--- Processed Data ---\n")

    print("X_train shape:", X_train_s.shape)
    print("Y_true_train shape:", Yc_train_s.shape)
    print("Y_noise_train shape:", Yn_train_s.shape)
    print()
    print("X_val shape:", X_val_s.shape)
    print("Y_true_val shape:", Yc_val_s.shape)
    print("Y_noise_val shape:", Yn_val_s.shape)

    print()

    print(f"X_train mean: {X_train_s.mean():.2f}")
    print(f"X_train std: {X_train_s.std():.2f}")
    print(f"Y_true_train mean: {Yc_train_s.mean():.2f}")
    print(f"Y_true_train std: {Yc_train_s.std():.2f}")
    print(f"Y_noise_train mean: {Yn_train_s.mean():.2f}")
    print(f"Y_noise_train std: {Yn_train_s.std():.2f}")
    print()
    print(f"X_val mean: {X_val_s.mean():.2f}")
    print(f"X_val std: {X_val_s.std():.2f}")
    print(f"Y_true_val mean: {Yc_val_s.mean():.2f}")
    print(f"Y_true_val std: {Yc_val_s.std():.2f}")
    print(f"Y_noise_val mean: {Yn_val_s.mean():.2f}")
    print(f"Y_noise_val std: {Yn_val_s.std():.2f}")

    np.save(out_dir / "X_train.npy", X_train_s)
    np.save(out_dir / "X_val.npy", X_val_s)

    np.save(out_dir / "Y_true_train.npy", Yc_train_s)
    np.save(out_dir / "Y_true_val.npy", Yc_val_s)
    
    np.save(out_dir / "Y_noise_train.npy", Yn_train_s)
    np.save(out_dir / "Y_noise_val.npy", Yn_val_s)

    np.save(out_dir / "scaler_mean.npy", np.array([mean], dtype=np.float64))   
    np.save(out_dir / "scaler_std.npy", np.array([std], dtype=np.float64))   

    print(f"Saved processed data to {out_dir}")
    return out_dir

if __name__ == "__main__":

    config_path = "configs/config_2layer.yaml"
    prepare_and_save(config_path)

    
