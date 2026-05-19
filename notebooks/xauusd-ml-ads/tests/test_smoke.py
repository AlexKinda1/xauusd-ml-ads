"""Smoke tests: verify the skeleton is importable and configs are valid YAML."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_imports() -> None:
    """All `src.*` packages can be imported without error."""
    import src  # noqa: F401
    from src.utils import config, logging, seed  # noqa: F401


def test_seed_is_deterministic() -> None:
    """Calling `set_global_seed` makes NumPy reproducible."""
    import numpy as np

    from src.utils.seed import set_global_seed

    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "yaml_path",
    [
        "config/data.yaml",
        "config/training.yaml",
        "config/models/xgboost.yaml",
        "config/models/random_forest.yaml",
        "config/models/cnn.yaml",
        "config/models/bigru.yaml",
        "config/models/chronos.yaml",
        "config/models/fincast.yaml",
        "config/models/baseline.yaml",
    ],
)
def test_config_yaml_loads(project_root: Path, yaml_path: str) -> None:
    """Every shipped config YAML parses to a non-empty dict."""
    from src.utils.config import load_yaml

    cfg = load_yaml(project_root / yaml_path)
    assert isinstance(cfg, dict)
    assert len(cfg) > 0
