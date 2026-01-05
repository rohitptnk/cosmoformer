from pathlib import Path
from typing import Optional, Tuple
import numpy  as np
import torch
from torch.utils.data import Dataset, random_split, Subset
from torch import Tensor

DEFAULT_DTYPE = torch.float32

def _load_scaler(proccessed_dir: Path) -> Tuple[float, float]:
    """loads the scaler .npy files and returns as python floats"""

    mean_path = proccessed_dir / "scaler_mean.npy"
    std_path = proccessed_dir / "scaler_std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Scaler files are not found in {proccessed_dir}")
    mean = float(np.load(mean_path)[0])
    std = float(np.load(std_path)[0])
    return mean, std

class ClDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str = "train", dtype: torch.dtype = DEFAULT_DTYPE):
        processed_dir = Path(processed_dir) 
        split = split.lower()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        # load X and Y (noisy and true) either 'train' or 'val' split
        X_path = processed_dir / f"X_{split}.npy"
        Y_true_path = processed_dir / f"Y_true_{split}.npy"
        Y_noise_path = processed_dir /f"Y_noise_{split}.npy"
        if not X_path.exists() or not Y_true_path.exists() or not Y_noise_path.exists():
            raise FileNotFoundError(f"Processed split files not found: {X_path}, {Y_true_path}, {Y_noise_path}")
        
        X = np.load(X_path)
        Y_true = np.load(Y_true_path)
        Y_noise = np.load(Y_noise_path)

        if X.shape != Y_true.shape or X.shape != Y_noise.shape:
            raise ValueError(
                f"Inconsistent shapes: X {X.shape}, "
                f"Y_true {Y_true.shape}, Y_noise {Y_noise.shape}")
        
        self.X = torch.from_numpy(X).to(dtype)
        self.Y_true = torch.from_numpy(Y_true).to(dtype)
        self.Y_noise = torch.from_numpy(Y_noise).to(dtype)

        # load the scalers
        mean, std = _load_scaler(processed_dir)
        self.mean = torch.tensor(mean, dtype=dtype)
        self.std = torch.tensor(std, dtype=dtype)

    def __len__(self) -> int:
        return self.X.shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return self.X[idx], (self.Y_true[idx], self.Y_noise[idx])
        
    def inverse_transform(self, arr: Tensor | np.ndarray) -> Tensor:
        if not torch.is_tensor(arr):
            arr = torch.tensor(arr, dtype=self.mean.dtype)
        return arr * self.std.to(arr.device) + self.mean.to(arr.device)
    
    def var_denormalize(self, var_norm: Tensor | np.ndarray) -> Tensor:
        """
        Given variance in normalized units (var_norm), convert to original units:
            var_orig = var_norm * std^2
        """
        if not torch.is_tensor(var_norm):
            var_norm = torch.tensor(var_norm, dtype=self.std.dtype)
        return var_norm * (self.std.to(var_norm.device)**2)
        

