"""Abstract base class for every model in this project.

Why an explicit ABC:
- The 7 model families have wildly different internals (statsmodels ARIMA,
  XGBoost Booster, sklearn estimators, PyTorch nn.Modules, HuggingFace TSFMs).
- The training, evaluation, and reporting layers must treat them uniformly.
- An ABC ensures every new model implements ``fit/predict/save/load`` with
  the same signature, so the orchestration code (Phase 5) does not branch
  on model type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

Task = Literal["regression", "classification"]


class ModelBase(ABC):
    """Common interface for all models (baseline → TSFM).

    Subclasses must implement :meth:`fit`, :meth:`predict`, :meth:`save`,
    :meth:`load`. Classification subclasses should additionally override
    :meth:`predict_proba` when probabilities are available.
    """

    #: Human-readable identifier used in MLflow runs and prediction filenames.
    name: str = "abstract"
    #: ``"regression"`` or ``"classification"``.
    task: Task

    def __init__(self, task: Task, **kwargs: Any) -> None:
        if task not in {"regression", "classification"}:
            raise ValueError(f"Unknown task: {task!r}")
        self.task = task
        self.params: dict[str, Any] = kwargs

    # ------------------------------------------------------------------ API
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "ModelBase":
        """Train on ``(X_train, y_train)``. ``X_val/y_val`` enable early stopping
        where supported. Return ``self`` for chaining."""

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Point predictions. For classification, returns labels ``{-1, 0, 1}``."""

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Class probabilities. Override in classification subclasses if available.

        Default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide predict_proba."
        )

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist the trained model to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "ModelBase":
        """Restore a previously :meth:`save`-d model."""

    # ----------------------------------------------------------- utilities
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(task={self.task!r}, name={self.name!r})"
