from pathlib import Path
from typing import Optional, Tuple
import numpy  as np
import torch
from torch.utils.data import Dataset, random_split, Subset
from torch import Tensor

DEFAULT_DTYPE = torch.float32

def _load_scaler(proccessed_dir: Path, name: str) -> Tuple[float, float]:
    """loads the scaler .npy files and returns as python floats"""

    mean_path = proccessed_dir / f"scaler_{name}_mean.npy"
    std_path = proccessed_dir / f"scaler_{name}_std.npy"
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
        mean_X, std_X = _load_scaler(processed_dir, "X")
        self.mean_X = torch.tensor(mean_X, dtype=dtype)
        self.std_X = torch.tensor(std_X, dtype=dtype)

        mean_Yc, std_Yc = _load_scaler(processed_dir, "Yc")
        self.mean_Yc = torch.tensor(mean_Yc, dtype=dtype)
        self.std_Yc = torch.tensor(std_Yc, dtype=dtype)

        mean_Yn, std_Yn = _load_scaler(processed_dir, "Yn")
        self.mean_Yn = torch.tensor(mean_Yn, dtype=dtype)
        self.std_Yn = torch.tensor(std_Yn, dtype=dtype)

    def __len__(self) -> int:
        return self.X.shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return self.X[idx], (self.Y_true[idx], self.Y_noise[idx])
        
    def inverse_transform_Yc(self, arr: Tensor | np.ndarray) -> Tensor:
        if not torch.is_tensor(arr):
            arr = torch.tensor(arr, dtype=self.mean_Yc.dtype)
        return arr * self.std_Yc.to(arr.device) + self.mean_Yc.to(arr.device)

    def inverse_transform_Yn(self, arr: Tensor | np.ndarray) -> Tensor:
        if not torch.is_tensor(arr):
            arr = torch.tensor(arr, dtype=self.mean_Yn.dtype)
        return arr * self.std_Yn.to(arr.device) + self.mean_Yn.to(arr.device)
    
    def var_denormalize_Yc(self, var_norm: Tensor | np.ndarray) -> Tensor:
        if not torch.is_tensor(var_norm):
            var_norm = torch.tensor(var_norm, dtype=self.std_Yc.dtype)
        return var_norm * (self.std_Yc.to(var_norm.device)**2)

    def var_denormalize_Yn(self, var_norm: Tensor | np.ndarray) -> Tensor:
        if not torch.is_tensor(var_norm):
            var_norm = torch.tensor(var_norm, dtype=self.std_Yn.dtype)
        return var_norm * (self.std_Yn.to(var_norm.device)**2)
        

