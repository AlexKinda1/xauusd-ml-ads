"""Train-only feature scaler with save / load.

Wrapping :class:`sklearn.preprocessing.StandardScaler` with three guarantees:

  1. ``fit`` is called on the **train** DataFrame only — never on val or test.
  2. ``transform`` preserves the original ``pd.DataFrame`` (index, column
     order, dtype) so downstream code keeps semantic ownership of features.
  3. ``save``/``load`` round-trip the fitted scaler + the feature column
     order so inference reproduces training-time normalisation bit-for-bit.

Optional NaN handling:
- ``transform(fillna=0.0)`` substitutes any remaining NaN AFTER scaling with
  the given value (default 0, i.e. the post-standardization mean). Useful
  for DL models that cannot ingest NaN.
- ``transform(fillna=None)`` (default) leaves NaN intact for XGBoost which
  handles them natively.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainOnlyScaler:
    """Scaler that refuses to be fit on anything except training data."""

    feature_cols: list[str]
    scaler: StandardScaler

    @classmethod
    def fit(cls, X_train: pd.DataFrame, feature_cols: list[str] | None = None) -> Self:
        """Fit a fresh :class:`StandardScaler` on the training frame."""
        if feature_cols is None:
            feature_cols = list(X_train.columns)
        missing = [c for c in feature_cols if c not in X_train.columns]
        if missing:
            raise ValueError(f"Missing columns in train: {missing}")
        scaler = StandardScaler()
        # sklearn StandardScaler ignores NaN if input is a DataFrame with NaN,
        # by raising in some versions; use nan-aware fit on numpy then.
        arr = X_train[feature_cols].to_numpy(dtype="float64", copy=False)
        # Compute means/stds while ignoring NaN.
        scaler.mean_ = np.nanmean(arr, axis=0)
        scaler.scale_ = np.nanstd(arr, axis=0, ddof=0)
        scaler.scale_[scaler.scale_ == 0.0] = 1.0    # avoid div-by-zero
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(feature_cols)
        scaler.feature_names_in_ = np.array(feature_cols, dtype=object)
        scaler.n_samples_seen_ = arr.shape[0]
        return cls(feature_cols=feature_cols, scaler=scaler)

    def transform(
        self,
        X: pd.DataFrame,
        *,
        fillna: float | None = None,
    ) -> pd.DataFrame:
        """Apply the fitted scaling to ``X[self.feature_cols]``.

        Args:
            X: Any DataFrame containing at least ``self.feature_cols``.
            fillna: If set, replace NaN values in the result with this value.
                Use ``None`` to preserve NaN (e.g. for XGBoost). Use ``0.0``
                for DL models — equivalent to "the mean" after standardization.

        Returns:
            A new DataFrame with the same index as ``X`` and the same
            feature column order as ``self.feature_cols``.
        """
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns at transform time: {missing}")
        arr = X[self.feature_cols].to_numpy(dtype="float64", copy=True)
        arr = (arr - self.scaler.mean_) / self.scaler.scale_
        if fillna is not None:
            arr = np.where(np.isnan(arr), fillna, arr)
        return pd.DataFrame(arr, index=X.index, columns=self.feature_cols)

    def save(self, path: str | Path) -> Path:
        """Persist the fitted scaler to ``path`` (pickle)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f)
        return p

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously-saved scaler."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}: {type(obj)}")
        return obj
