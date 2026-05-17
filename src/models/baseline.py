"""Baseline models: Naïve and AR(p) on log-returns.

Two baselines per task (4 models total):

  - **Regression**
    - :class:`NaiveZeroRegressor` : predicts ``y_reg = 0`` (random-walk null).
      The bar against which every ML/DL model must demonstrate added value.
    - :class:`ARRegressor` : AR(p) on the 1-bar log-return series, with
      vectorised closed-form ``h``-step forecast. Order selected by AIC on
      train among a small grid.

  - **Classification**
    - :class:`MajorityClassifier` : predicts the most frequent train class.
    - :class:`ARClassifier` : derives a label from the AR regression
      prediction using the same ``0.5 × σ × √h`` threshold as the target.

The AR baselines deliberately avoid the costly per-row ``ARIMA.apply()``
that ``statsmodels`` requires when predicting on a long series. We instead
fit AR(p) once on train and reuse the closed-form recursion that exists for
the conditional mean of an AR process.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.models.base import ModelBase, Task


# ---------------------------------------------------------------------------
# Naïve baselines
# ---------------------------------------------------------------------------


class NaiveZeroRegressor(ModelBase):
    """Predict ``y_reg = 0`` always.

    Equivalent to assuming the asset follows a martingale with no drift.
    This is the academic gold-standard null hypothesis for return prediction.
    """

    name = "naive_zero"

    def __init__(self) -> None:
        super().__init__(task="regression")

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "NaiveZeroRegressor":  # noqa: D401, ARG002
        return self

    def predict(self, X) -> np.ndarray:
        return np.zeros(len(X), dtype="float64")

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b""); return p

    @classmethod
    def load(cls, path: str | Path) -> "NaiveZeroRegressor":  # noqa: ARG003
        return cls()


class MajorityClassifier(ModelBase):
    """Predict the most frequent training-set class for every row."""

    name = "majority"

    def __init__(self) -> None:
        super().__init__(task="classification")
        self.majority_class_: int | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "MajorityClassifier":  # noqa: ARG002
        y = pd.Series(y_train).dropna().astype(int)
        self.majority_class_ = int(y.mode().iloc[0])
        return self

    def predict(self, X) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("MajorityClassifier must be fit before predict.")
        return np.full(len(X), self.majority_class_, dtype="int64")

    def predict_proba(self, X) -> np.ndarray:
        # One-hot probability concentrated on majority class.
        n = len(X)
        probs = np.zeros((n, 3), dtype="float64")
        col = {-1: 0, 0: 1, 1: 2}[int(self.majority_class_ or 0)]
        probs[:, col] = 1.0
        return probs

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump({"majority_class_": self.majority_class_}, f)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "MajorityClassifier":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        m = cls(); m.majority_class_ = state["majority_class_"]; return m


# ---------------------------------------------------------------------------
# AR(p) baseline
# ---------------------------------------------------------------------------


def _fit_ar_select_order(returns: np.ndarray, p_grid: tuple[int, ...] = (1, 2, 3, 5)) -> tuple[int, np.ndarray, float]:
    """Fit AR(p) for each ``p`` in ``p_grid`` and return the best by AIC.

    Returns ``(best_p, ar_coefs, intercept)``. ``ar_coefs[i] = phi_{i+1}``.
    """
    best = None
    for p in p_grid:
        try:
            fit = ARIMA(returns, order=(p, 0, 0), trend="c").fit(method_kwargs={"warn_convergence": False})
        except Exception:  # noqa: BLE001
            continue
        aic = float(fit.aic)
        if best is None or aic < best[3]:
            # statsmodels params can be either a numpy array (ordered: const, ar.L1..ar.Lp)
            # or a pandas Series indexed by name. Handle both robustly.
            params = fit.params
            if hasattr(params, "loc"):  # pandas
                names = list(params.index)
                values = params.to_numpy()
            else:  # numpy
                names = list(getattr(fit, "param_names", [])) or [f"p{i}" for i in range(len(params))]
                values = np.asarray(params, dtype="float64")
            intercept = 0.0
            ar_coefs = np.zeros(p, dtype="float64")
            for nm, v in zip(names, values):
                if nm == "const" or nm == "intercept":
                    intercept = float(v)
                elif nm.startswith("ar.L"):
                    idx = int(nm[len("ar.L"):]) - 1
                    if 0 <= idx < p:
                        ar_coefs[idx] = float(v)
            best = (p, ar_coefs, intercept, aic)
    if best is None:
        raise RuntimeError("All AR fits failed.")
    return best[0], best[1], best[2]


def _h_step_sum_forecast(
    history: np.ndarray, ar_coefs: np.ndarray, intercept: float, h: int
) -> float:
    """Sum of the next ``h`` conditional-mean AR forecasts given ``history``.

    Args:
        history: 1D array; the last ``p = len(ar_coefs)`` entries are used.
        ar_coefs: ``[phi_1, ..., phi_p]``.
        intercept: ``c`` (the AR mean intercept).
        h: Number of steps to forecast.

    Returns:
        ``sum_{i=1..h} E[y_{t+i} | history]`` — the predicted horizon log-return
        when the AR is fit on 1-bar log-returns.
    """
    p = len(ar_coefs)
    state = list(history[-p:]) if p > 0 else []
    total = 0.0
    for _ in range(h):
        nxt = intercept
        for j in range(p):
            nxt += ar_coefs[j] * state[-(j + 1)]
        total += nxt
        state.append(nxt)
    return total


class ARRegressor(ModelBase):
    """AR(p) on 1-bar log-returns, vectorised h-step forecast.

    The model treats ``X`` as auxiliary metadata — only the close-price column
    matters. We retrieve 1-bar log-returns ourselves so the user does not need
    to align extra series.
    """

    name = "ar"

    def __init__(self, horizon: int = 24, p_grid: tuple[int, ...] = (1, 2, 3, 5)) -> None:
        super().__init__(task="regression", horizon=horizon, p_grid=p_grid)
        self.horizon = horizon
        self.p_grid = p_grid
        self.order_p_: int | None = None
        self.ar_coefs_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ARRegressor":  # noqa: ARG002
        close = self._extract_close(X_train)
        returns = np.diff(np.log(close.values))
        returns = returns[~np.isnan(returns)]
        self.order_p_, self.ar_coefs_, self.intercept_ = _fit_ar_select_order(
            returns, p_grid=self.p_grid
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.ar_coefs_ is None:
            raise RuntimeError("ARRegressor must be fit before predict.")
        close = self._extract_close(X)
        log_close = np.log(close.values)
        returns = np.diff(log_close, prepend=log_close[0])  # 0 at t=0
        p = len(self.ar_coefs_)
        preds = np.empty(len(X), dtype="float64")
        for i in range(len(X)):
            history = returns[max(0, i - p + 1) : i + 1]
            if len(history) < p:
                history = np.concatenate([np.zeros(p - len(history)), history])
            preds[i] = _h_step_sum_forecast(history, self.ar_coefs_, float(self.intercept_), self.horizon)
        return preds

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(
                {
                    "order_p_": self.order_p_,
                    "ar_coefs_": self.ar_coefs_,
                    "intercept_": self.intercept_,
                    "horizon": self.horizon,
                    "p_grid": self.p_grid,
                },
                f,
            )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ARRegressor":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        m = cls(horizon=state["horizon"], p_grid=state["p_grid"])
        m.order_p_ = state["order_p_"]
        m.ar_coefs_ = state["ar_coefs_"]
        m.intercept_ = state["intercept_"]
        return m

    @staticmethod
    def _extract_close(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            if "close" not in X.columns:
                raise ValueError("ARRegressor expects a 'close' column in X.")
            return X["close"]
        raise TypeError("ARRegressor requires a DataFrame input with a 'close' column.")


class ARClassifier(ModelBase):
    """Apply the AR(p) forecast and threshold the result with ``0.5 × σ × √h``."""

    name = "ar"

    def __init__(self, horizon: int = 24, vol_window: int = 24, threshold_factor: float = 0.5,
                 p_grid: tuple[int, ...] = (1, 2, 3, 5)) -> None:
        super().__init__(
            task="classification", horizon=horizon, vol_window=vol_window,
            threshold_factor=threshold_factor, p_grid=p_grid,
        )
        self.horizon = horizon
        self.vol_window = vol_window
        self.threshold_factor = threshold_factor
        self.regressor = ARRegressor(horizon=horizon, p_grid=p_grid)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ARClassifier":  # noqa: ARG002
        # Use a dummy regression target — the regressor only needs the close column.
        self.regressor.fit(X_train, y_train=None)
        return self

    def predict(self, X) -> np.ndarray:
        y_reg = self.regressor.predict(X)
        close = ARRegressor._extract_close(X).values
        log_close = np.log(close)
        ret_1 = np.diff(log_close, prepend=log_close[0])
        ret_1[0] = 0.0
        # Rolling past vol with lag=1 (same convention as target construction).
        vol = pd.Series(ret_1).rolling(self.vol_window, min_periods=self.vol_window).std().shift(1).values
        threshold = self.threshold_factor * vol * np.sqrt(self.horizon)
        labels = np.zeros(len(X), dtype="int64")
        labels[(y_reg > threshold) & ~np.isnan(threshold)] = 1
        labels[(y_reg < -threshold) & ~np.isnan(threshold)] = -1
        # Where threshold is NaN (warm-up), default to 0.
        labels[np.isnan(threshold)] = 0
        return labels

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(
                {
                    "horizon": self.horizon, "vol_window": self.vol_window,
                    "threshold_factor": self.threshold_factor,
                    "regressor_state": {
                        "order_p_": self.regressor.order_p_,
                        "ar_coefs_": self.regressor.ar_coefs_,
                        "intercept_": self.regressor.intercept_,
                        "horizon": self.regressor.horizon,
                        "p_grid": self.regressor.p_grid,
                    },
                },
                f,
            )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ARClassifier":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        m = cls(
            horizon=state["horizon"], vol_window=state["vol_window"],
            threshold_factor=state["threshold_factor"],
            p_grid=state["regressor_state"]["p_grid"],
        )
        m.regressor.order_p_ = state["regressor_state"]["order_p_"]
        m.regressor.ar_coefs_ = state["regressor_state"]["ar_coefs_"]
        m.regressor.intercept_ = state["regressor_state"]["intercept_"]
        return m
