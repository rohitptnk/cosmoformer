from pathlib import Path
import yaml
from typing import Dict, Any, Union


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    try:
        from .config_schema import Config as ConfigModel
        from pydantic import ValidationError
    except ImportError as e:
        raise ImportError(
            "pydantic is required for config validation. Install it with 'pip install pydantic>=1.10,<2'"
        ) from e

    try:
        cfg_model = ConfigModel.model_validate(cfg)
    except ValidationError as e:
        raise ValueError(f"Config validation error:\n{e}") from e

    return cfg_model.model_dump()