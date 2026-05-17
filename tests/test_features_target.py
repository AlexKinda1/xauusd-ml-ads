"""Tests for the regression and classification targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import target as tgt


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC"))


def test_regression_target_formula() -> None:
    close = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    y = tgt.regression_target(close, horizon=2)
    # y[0] = log(102/100), y[1] = log(103/101), ...
    assert np.isclose(y.iloc[0], np.log(102 / 100))
    assert np.isclose(y.iloc[1], np.log(103 / 101))
    # Last 2 rows are NaN (no future)
    assert y.iloc[-2:].isna().all()


def test_realized_vol_past_does_not_use_t() -> None:
    """Vol at t must depend only on returns up to t-1."""
    close = _series([100, 101, 102, 103, 104, 105, 200])  # spike at t=6
    vol = tgt.realized_vol_past(close, window=3, lag=1)
    # vol.iloc[6] is computed from returns 4->5 and 5->6 ... but lag=1 shifts
    # → vol at t=6 actually used returns 3->4 and 4->5 (no spike).
    # So vol.iloc[6] should NOT explode due to the spike at t=6.
    assert vol.iloc[6] < 1.0   # no spike contamination


def test_classification_target_three_classes() -> None:
    np.random.seed(0)
    close = pd.Series(100 + np.cumsum(np.random.normal(0, 0.5, 500)),
                      index=pd.date_range("2024-01-01", periods=500, freq="1h", tz="UTC"))
    y_reg = tgt.regression_target(close, horizon=24)
    y_clf = tgt.classification_target(y_reg, close, horizon=24, vol_window=24, threshold_factor=0.5)
    counts = y_clf.dropna().value_counts()
    # All three classes present
    assert set(counts.index) == {-1.0, 0.0, 1.0}


def test_classification_target_threshold_factor_zero_collapses_to_sign() -> None:
    np.random.seed(1)
    close = pd.Series(100 + np.cumsum(np.random.normal(0, 0.5, 500)),
                      index=pd.date_range("2024-01-01", periods=500, freq="1h", tz="UTC"))
    y_reg = tgt.regression_target(close, horizon=24)
    y_clf = tgt.classification_target(y_reg, close, horizon=24, vol_window=24, threshold_factor=0.0)
    # With factor=0, no neutral zone except where y_reg == 0 exactly (measure 0)
    n_neutral = (y_clf == 0).sum()
    assert n_neutral <= 1
