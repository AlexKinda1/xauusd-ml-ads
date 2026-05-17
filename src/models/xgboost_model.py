"""XGBoost wrappers for regression and 3-class classification.

Native XGBoost handling of NaN values means the tabular features can be
passed in as-is — fear_greed missing pre-2018 is simply treated as a
"learnable" branching path during tree construction.

Class labels in the project are ``{-1, 0, 1}``. XGBoost needs ``{0, 1, ...}``.
We remap on ingress and on egress.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import ModelBase

# Label encoding
_CLF_TO_XGB = {-1: 0, 0: 1, 1: 2}
_XGB_TO_CLF = {v: k for k, v in _CLF_TO_XGB.items()}


def _to_X(df_or_array: pd.DataFrame | np.ndarray, feature_cols: list[str] | None) -> np.ndarray | pd.DataFrame:
    """Subset a DataFrame to ``feature_cols`` or return the raw array."""
    if isinstance(df_or_array, pd.DataFrame):
        return df_or_array[feature_cols] if feature_cols else df_or_array
    return df_or_array


class XGBoostRegressor(ModelBase):
    """XGBoost gradient-boosted trees for log-return regression."""

    name = "xgboost"

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(task="regression", **kwargs)
        self.feature_cols = feature_cols
        self.model_params = dict(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            tree_method="hist",
        )
        self.params.update(self.model_params)
        self.booster_: xgb.XGBRegressor | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBoostRegressor":
        Xt = _to_X(X_train, self.feature_cols)
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(_to_X(X_val, self.feature_cols), y_val)]
        self.booster_ = xgb.XGBRegressor(**self.model_params)
        self.booster_.fit(Xt, y_train, eval_set=eval_set or None, verbose=False)
        return self

    def predict(self, X) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("XGBoostRegressor must be fit before predict.")
        return self.booster_.predict(_to_X(X, self.feature_cols))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.booster_.feature_importances_   # type: ignore[union-attr]

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(p.with_suffix(".json")))
        with p.with_suffix(".meta.pkl").open("wb") as f:
            pickle.dump({"feature_cols": self.feature_cols, "params": self.model_params}, f)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostRegressor":
        p = Path(path)
        with p.with_suffix(".meta.pkl").open("rb") as f:
            meta = pickle.load(f)
        m = cls(feature_cols=meta["feature_cols"], **meta["params"])
        m.booster_ = xgb.XGBRegressor()
        m.booster_.load_model(str(p.with_suffix(".json")))
        return m


class XGBoostClassifier(ModelBase):
    """XGBoost multi-class classifier for ternary direction labels."""

    name = "xgboost"

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(task="classification", **kwargs)
        self.feature_cols = feature_cols
        self.model_params = dict(
            objective="multi:softprob",
            num_class=3,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            tree_method="hist",
            eval_metric="mlogloss",
        )
        self.params.update(self.model_params)
        self.booster_: xgb.XGBClassifier | None = None

    @staticmethod
    def _encode(y: np.ndarray) -> np.ndarray:
        return np.vectorize(_CLF_TO_XGB.get)(y.astype(int))

    @staticmethod
    def _decode(y: np.ndarray) -> np.ndarray:
        return np.vectorize(_XGB_TO_CLF.get)(y.astype(int))

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBoostClassifier":
        Xt = _to_X(X_train, self.feature_cols)
        yt = self._encode(np.asarray(y_train))
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(_to_X(X_val, self.feature_cols), self._encode(np.asarray(y_val)))]
        self.booster_ = xgb.XGBClassifier(**self.model_params)
        self.booster_.fit(Xt, yt, eval_set=eval_set or None, verbose=False)
        return self

    def predict(self, X) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("XGBoostClassifier must be fit before predict.")
        out = self.booster_.predict(_to_X(X, self.feature_cols))
        return self._decode(out)

    def predict_proba(self, X) -> np.ndarray:
        return self.booster_.predict_proba(_to_X(X, self.feature_cols))   # type: ignore[union-attr]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.booster_.feature_importances_   # type: ignore[union-attr]

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(p.with_suffix(".json")))
        with p.with_suffix(".meta.pkl").open("wb") as f:
            pickle.dump({"feature_cols": self.feature_cols, "params": self.model_params}, f)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostClassifier":
        p = Path(path)
        with p.with_suffix(".meta.pkl").open("rb") as f:
            meta = pickle.load(f)
        m = cls(feature_cols=meta["feature_cols"], **meta["params"])
        m.booster_ = xgb.XGBClassifier()
        m.booster_.load_model(str(p.with_suffix(".json")))
        return m
