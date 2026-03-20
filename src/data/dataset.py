from pathlib import Path
from typing import Optional, Tuple, Union
import numpy  as np
import torch
from torch.utils.data import Dataset, random_split, Subset
from torch import Tensor

DEFAULT_DTYPE = torch.float32

def _load_scaler(proccessed_dir: Path, key: str) -> Tuple[float, float]:
    """loads the scaler .npy files for a specific signal and returns as python floats"""

    mean_path = proccessed_dir / f"{key}_scaler_mean.npy"
    std_path = proccessed_dir / f"{key}_scaler_std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Scaler files for {key} are not found in {proccessed_dir}")
    mean = float(np.load(mean_path)[0])
    std = float(np.load(std_path)[0])
    return mean, std

class ClDataset(Dataset):
    def __init__(self, processed_dir: Union[str, Path], split: str = "train", dtype: torch.dtype = DEFAULT_DTYPE):
        processed_dir = Path(processed_dir) 
        split = split.lower()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        # Files to load
        files = {
            "X1": f"X1_{split}.npy",
            "X2": f"X2_{split}.npy",
            "X3": f"X3_{split}.npy",
            "X4": f"X4_{split}.npy",
            "Y_true": f"Y_true_{split}.npy",
            "Y_freq1": f"Y_freq1_{split}.npy",
            "Y_freq2": f"Y_freq2_{split}.npy",
            "Y_freq3": f"Y_freq3_{split}.npy",
            "Y_freq4": f"Y_freq4_{split}.npy"
        }

        self.data = {}
        for key, fname in files.items():
            path = processed_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Processed split file not found: {path}")
            arr = np.load(path)
            self.data[key] = torch.from_numpy(arr).to(dtype)

        # check shapes
        ref_shape = self.data["X1"].shape
        for key, tensor in self.data.items():
            if tensor.shape != ref_shape:
                raise ValueError(f"Inconsistent shapes in {split} split for {key}: {tensor.shape} vs {ref_shape}")

        # load the scalers
        self.scalers = {}
        for key in files.keys():
            mean, std = _load_scaler(processed_dir, key)
            self.scalers[key] = {
                "mean": torch.tensor(mean, dtype=dtype),
                "std": torch.tensor(std, dtype=dtype)
            }

    def __len__(self) -> int:
        return self.data["X1"].shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        return (self.data["X1"][idx], self.data["X2"][idx], self.data["X3"][idx], self.data["X4"][idx]), \
               (self.data["Y_true"][idx], self.data["Y_freq1"][idx], self.data["Y_freq2"][idx], self.data["Y_freq3"][idx], self.data["Y_freq4"][idx])
        
    def inverse_transform(self, arr: Union[Tensor, np.ndarray], key: str) -> Tensor:
        if not torch.is_tensor(arr):
            arr = torch.tensor(arr, dtype=self.scalers[key]["mean"].dtype)
        
        mean = self.scalers[key]["mean"].to(arr.device)
        std = self.scalers[key]["std"].to(arr.device)
        return arr * std + mean
    
    def var_denormalize(self, var_norm: Union[Tensor, np.ndarray], key: str) -> Tensor:
        """
        Given variance in normalized units (var_norm), convert to original units:
            var_orig = var_norm * std^2
        """
        if not torch.is_tensor(var_norm):
            var_norm = torch.tensor(var_norm, dtype=self.scalers[key]["std"].dtype)
            
        std = self.scalers[key]["std"].to(var_norm.device)
        return var_norm * (std**2)
        

