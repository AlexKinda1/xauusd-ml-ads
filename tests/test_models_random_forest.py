"""Tests for the Random Forest wrappers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.random_forest import (
    RandomForestClassifierModel,
    RandomForestRegressorModel,
)


@pytest.fixture
def synthetic() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rs = np.random.RandomState(0)
    X = pd.DataFrame({f"f{i}": rs.normal(0, 1, n) for i in range(8)}, index=idx)
    y_reg = (0.3 * X["f0"] - 0.2 * X["f1"] + rs.normal(0, 0.1, n)).values
    score = 0.5 * X["f0"] + 0.3 * X["f3"] - 0.4 * X["f5"] + rs.normal(0, 0.3, n)
    y_clf = np.where(score > 0.3, 1, np.where(score < -0.3, -1, 0))
    full = X.assign(y_reg=y_reg, y_clf=y_clf)
    return full.iloc[:400], full.iloc[400:], list(X.columns)


def test_rf_regressor_fits_and_beats_zero(synthetic) -> None:
    train, val, feats = synthetic
    m = RandomForestRegressorModel(feature_cols=feats, n_estimators=80).fit(
        train, train["y_reg"].values
    )
    pred = m.predict(val)
    mse_model = float(np.mean((val["y_reg"].values - pred) ** 2))
    mse_zero = float(np.mean(val["y_reg"].values ** 2))
    assert mse_model < mse_zero * 0.7


def test_rf_classifier_ternary(synthetic) -> None:
    train, val, feats = synthetic
    m = RandomForestClassifierModel(feature_cols=feats, n_estimators=80).fit(
        train, train["y_clf"].values
    )
    pred = m.predict(val)
    assert set(np.unique(pred)).issubset({-1, 0, 1})
    proba = m.predict_proba(val)
    assert proba.shape == (len(val), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_rf_imputes_nan_with_train_median(synthetic) -> None:
    train, val, feats = synthetic
    train_with_nan = train.copy()
    train_with_nan.iloc[:50, train_with_nan.columns.get_loc("f0")] = np.nan
    m = RandomForestRegressorModel(feature_cols=feats, n_estimators=20).fit(
        train_with_nan, train_with_nan["y_reg"].values
    )
    val_with_nan = val.copy()
    val_with_nan.iloc[0, val_with_nan.columns.get_loc("f0")] = np.nan
    pred = m.predict(val_with_nan)
    assert np.isfinite(pred).all()


def test_rf_save_load_roundtrip(synthetic) -> None:
    train, val, feats = synthetic
    with tempfile.TemporaryDirectory() as tmp:
        for ctor, target in [
            (RandomForestRegressorModel, "y_reg"),
            (RandomForestClassifierModel, "y_clf"),
        ]:
            m = ctor(feature_cols=feats, n_estimators=20, n_jobs=1).fit(
                train, train[target].values
            )
            before = m.predict(val)
            p = Path(tmp) / f"{ctor.__name__}.pkl"
            m.save(p)
            reloaded = ctor.load(p)
            after = reloaded.predict(val)
            # n_jobs=1 ensures deterministic summation order across pickle.
            np.testing.assert_allclose(before, after, rtol=1e-12)
