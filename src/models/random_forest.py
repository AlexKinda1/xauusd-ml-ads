"""Random Forest wrappers for regression and 3-class classification.

Unlike XGBoost, sklearn's RandomForest cannot handle NaN values natively.
We impute the median of the training set for any feature with missing
values, preserving the median for inference via attribute ``_train_median``.

The ternary classification label encoding follows the same {-1, 0, 1}
convention used throughout the project. sklearn accepts those integer
labels directly, so no encoding/decoding is needed for the classifier.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models.base import ModelBase


def _to_X(df_or_array: pd.DataFrame | np.ndarray, feature_cols: list[str] | None):
    if isinstance(df_or_array, pd.DataFrame):
        return df_or_array[feature_cols] if feature_cols else df_or_array
    return df_or_array


def _impute_nan(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Replace NaN by the per-column median computed at fit time."""
    return X.fillna(medians)


class RandomForestRegressorModel(ModelBase):
    """Random Forest for log-return regression."""

    name = "random_forest"

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str | float = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(task="regression", **kwargs)
        self.feature_cols = feature_cols
        self.model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.params.update(self.model_params)
        self.estimator_: RandomForestRegressor | None = None
        self._train_median: pd.Series | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "RandomForestRegressorModel":  # noqa: ARG002
        Xt = _to_X(X_train, self.feature_cols).copy()
        self._train_median = Xt.median(numeric_only=True)
        Xt = _impute_nan(Xt, self._train_median)
        self.estimator_ = RandomForestRegressor(**self.model_params)
        self.estimator_.fit(Xt, y_train)
        return self

    def predict(self, X) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("Model not fit.")
        Xv = _impute_nan(_to_X(X, self.feature_cols), self._train_median)
        return self.estimator_.predict(Xv)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator_.feature_importances_   # type: ignore[union-attr]

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(
                {
                    "kind": "rf_regressor",
                    "feature_cols": self.feature_cols,
                    "params": self.model_params,
                    "estimator": self.estimator_,
                    "train_median": self._train_median,
                },
                f,
            )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestRegressorModel":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        m = cls(feature_cols=state["feature_cols"], **state["params"])
        m.estimator_ = state["estimator"]
        m._train_median = state["train_median"]
        return m


class RandomForestClassifierModel(ModelBase):
    """Random Forest classifier for ternary direction labels."""

    name = "random_forest"

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str | float = "sqrt",
        class_weight: str | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(task="classification", **kwargs)
        self.feature_cols = feature_cols
        self.model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.params.update(self.model_params)
        self.estimator_: RandomForestClassifier | None = None
        self._train_median: pd.Series | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "RandomForestClassifierModel":  # noqa: ARG002
        Xt = _to_X(X_train, self.feature_cols).copy()
        self._train_median = Xt.median(numeric_only=True)
        Xt = _impute_nan(Xt, self._train_median)
        self.estimator_ = RandomForestClassifier(**self.model_params)
        self.estimator_.fit(Xt, np.asarray(y_train).astype(int))
        return self

    def predict(self, X) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("Model not fit.")
        Xv = _impute_nan(_to_X(X, self.feature_cols), self._train_median)
        return self.estimator_.predict(Xv)

    def predict_proba(self, X) -> np.ndarray:
        Xv = _impute_nan(_to_X(X, self.feature_cols), self._train_median)
        # Re-order columns so they match CLASS_LABELS (-1, 0, 1).
        proba = self.estimator_.predict_proba(Xv)   # type: ignore[union-attr]
        classes = list(self.estimator_.classes_)   # type: ignore[union-attr]
        target_order = [-1, 0, 1]
        idx = [classes.index(c) if c in classes else -1 for c in target_order]
        out = np.zeros((proba.shape[0], 3), dtype="float64")
        for col, src in enumerate(idx):
            if src >= 0:
                out[:, col] = proba[:, src]
        return out

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator_.feature_importances_   # type: ignore[union-attr]

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(
                {
                    "kind": "rf_classifier",
                    "feature_cols": self.feature_cols,
                    "params": self.model_params,
                    "estimator": self.estimator_,
                    "train_median": self._train_median,
                },
                f,
            )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestClassifierModel":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        m = cls(feature_cols=state["feature_cols"], **state["params"])
        m.estimator_ = state["estimator"]
        m._train_median = state["train_median"]
        return m
