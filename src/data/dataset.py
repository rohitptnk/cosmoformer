from pathlib import Path
from typing import Optional, Tuple, Union
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
    def __init__(self, processed_dir: Union[str, Path], split: str = "train", dtype: torch.dtype = DEFAULT_DTYPE):
        processed_dir = Path(processed_dir) 
        split = split.lower()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        # Files to load
        files = {
            "X1": f"X1_{split}.npy",
            "X2": f"X2_{split}.npy",
            "Y_true": f"Y_true_{split}.npy",
            "Y_fg1": f"Y_fg1_{split}.npy",
            "Y_fg2": f"Y_fg2_{split}.npy"
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
        mean, std = _load_scaler(processed_dir)
        self.mean = torch.tensor(mean, dtype=dtype)
        self.std = torch.tensor(std, dtype=dtype)

    def __len__(self) -> int:
        return self.data["X1"].shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        return (self.data["X1"][idx], self.data["X2"][idx]), \
               (self.data["Y_true"][idx], self.data["Y_fg1"][idx], self.data["Y_fg2"][idx])
        
    def inverse_transform(self, arr: Union[Tensor, np.ndarray]) -> Tensor:
        if not torch.is_tensor(arr):
            arr = torch.tensor(arr, dtype=self.mean.dtype)
        return arr * self.std.to(arr.device) + self.mean.to(arr.device)
    
    def var_denormalize(self, var_norm: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Given variance in normalized units (var_norm), convert to original units:
            var_orig = var_norm * std^2
        """
        if not torch.is_tensor(var_norm):
            var_norm = torch.tensor(var_norm, dtype=self.std.dtype)
        return var_norm * (self.std.to(var_norm.device)**2)
        

