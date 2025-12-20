from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    return cfg