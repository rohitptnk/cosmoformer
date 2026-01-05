import argparse
from pathlib import Path
import numpy as np

def prepare_and_save(raw_dir="data/raw", out_dir="data/processed", train_frac=0.8, seed=42, eps=1e-10):
    raw_dir=Path(raw_dir)
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(raw_dir / "mixed_cls_64_1k.npy") # mixed = clean + noise
    Y_clean = np.load(raw_dir / "true_cls_64_1k.npy")
    Y_noise = np.load(raw_dir / "noise_cls_64_1k.npy")

    if not (X.shape == Y_clean.shape == Y_noise.shape):
        raise ValueError(
            f"Inconsistent shapes: mixed {X.shape}, "
            f"clean {Y_clean.shape}, {Y_noise.shape}"
        )

    # random 80-20 train-test split
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = int(train_frac * N)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train = X[train_idx]
    Yc_train = Y_clean[train_idx]
    Yn_train = Y_noise[train_idx]

    X_val = X[val_idx]
    Yc_val = Y_clean[val_idx]
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
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, help="folder with noisy_cls_50k.npy and true_cls_50k.npy", default="data/raw")
    p.add_argument("--out_dir", type=str, default="data/processed")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    
    args = p.parse_args()  
    prepare_and_save(args.raw_dir, args.out_dir, args.train_frac, args.seed)

    
