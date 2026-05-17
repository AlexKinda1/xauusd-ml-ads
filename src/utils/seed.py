"""Global seed fixing for reproducibility.

Covers Python's `random`, NumPy, PyTorch (CPU + CUDA), and cuDNN determinism.
Call `set_global_seed(seed)` at the start of every entry point script.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Fix all sources of randomness.

    Args:
        seed: The integer seed to apply across libraries.
        deterministic_torch: If True, also forces cuDNN deterministic mode.
            Slightly slower but guarantees bit-reproducible PyTorch ops.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
