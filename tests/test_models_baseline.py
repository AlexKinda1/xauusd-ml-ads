"""Tests for the four baseline models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import (
    ARClassifier,
    ARRegressor,
    MajorityClassifier,
    NaiveZeroRegressor,
    _h_step_sum_forecast,
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    n = 1000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rs = np.random.RandomState(0)
    log_close = np.cumsum(rs.normal(0, 0.001, n))
    close = 2000 * np.exp(log_close)
    y_reg = np.log(np.roll(close, -24) / close)
    y_reg[-24:] = np.nan
    df = pd.DataFrame({"close": close, "y_reg": y_reg, "f1": rs.normal(0, 1, n)}, index=idx)
    return df


def test_naive_zero_predicts_zeros(small_df: pd.DataFrame) -> None:
    m = NaiveZeroRegressor().fit(small_df, small_df["y_reg"].values)
    out = m.predict(small_df)
    assert (out == 0).all()
    assert out.dtype == np.float64


def test_majority_classifier_constant(small_df: pd.DataFrame) -> None:
    labels = np.array([0, 0, 0, 1, -1, 0, 0, 0] * (len(small_df) // 8))[: len(small_df)]
    m = MajorityClassifier().fit(small_df, labels)
    out = m.predict(small_df)
    assert (out == 0).all()
    # predict_proba returns one-hot on majority class
    probs = m.predict_proba(small_df)
    assert probs.shape == (len(small_df), 3)
    assert (probs.sum(axis=1) == 1).all()


def test_h_step_sum_forecast_ar1_closed_form() -> None:
    # AR(1) with phi=0.5, c=0: forecast sum over h steps from y_t.
    # E[y_{t+i}] = phi^i * y_t. Sum = phi(1-phi^h)/(1-phi) * y_t.
    phi, h, y_t = 0.5, 24, 1.0
    expected = phi * (1 - phi**h) / (1 - phi) * y_t
    got = _h_step_sum_forecast(np.array([y_t]), np.array([phi]), 0.0, h)
    assert np.isclose(got, expected)


def test_ar_regressor_fits_and_predicts(small_df: pd.DataFrame) -> None:
    m = ARRegressor(horizon=24).fit(small_df, small_df["y_reg"].values)
    preds = m.predict(small_df)
    assert preds.shape == (len(small_df),)
    assert np.isfinite(preds).all()
    assert m.order_p_ in {1, 2, 3, 5}


def test_ar_classifier_labels_in_set(small_df: pd.DataFrame) -> None:
    m = ARClassifier(horizon=24).fit(small_df, np.zeros(len(small_df)))
    preds = m.predict(small_df)
    assert set(np.unique(preds)).issubset({-1, 0, 1})


def test_save_load_roundtrip_all_baselines(small_df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for ctor, fit_y in [
            (lambda: NaiveZeroRegressor(), small_df["y_reg"].values),
            (lambda: MajorityClassifier(), np.zeros(len(small_df))),
            (lambda: ARRegressor(horizon=24), small_df["y_reg"].values),
            (lambda: ARClassifier(horizon=24), np.zeros(len(small_df))),
        ]:
            model = ctor()
            model.fit(small_df, fit_y)
            before = model.predict(small_df)
            p = tmp / f"{model.name}_{model.task}.pkl"
            model.save(p)
            reloaded = model.__class__.load(p)
            after = reloaded.predict(small_df)
            np.testing.assert_array_equal(before, after)
