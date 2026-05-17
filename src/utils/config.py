"""YAML config loading utilities.

Configs are split across `config/data.yaml`, `config/training.yaml`, and
`config/models/<model>.yaml`. Loaders return plain dicts; downstream code is
free to wrap them in dataclasses where it improves clarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Args:
        path: Path to the YAML file. Relative paths resolve against the project
            root.

    Returns:
        Parsed YAML as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_config() -> dict[str, Any]:
    """Load `config/data.yaml`."""
    return load_yaml(CONFIG_DIR / "data.yaml")


def load_training_config() -> dict[str, Any]:
    """Load `config/training.yaml`."""
    return load_yaml(CONFIG_DIR / "training.yaml")


def load_model_config(model_name: str) -> dict[str, Any]:
    """Load `config/models/<model_name>.yaml`."""
    return load_yaml(CONFIG_DIR / "models" / f"{model_name}.yaml")
