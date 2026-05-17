"""Tests for the XGBoost regression / classification wrappers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import (
    _CLF_TO_XGB,
    _XGB_TO_CLF,
    XGBoostClassifier,
    XGBoostRegressor,
)


@pytest.fixture
def synthetic() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rs = np.random.RandomState(0)
    X = pd.DataFrame(
        {f"f{i}": rs.normal(0, 1, n) for i in range(8)},
        index=idx,
    )
    # Linear-ish target so the booster has signal to learn.
    y_reg = (0.3 * X["f0"] - 0.2 * X["f1"] + 0.1 * X["f2"] + rs.normal(0, 0.1, n)).values
    # Ternary classification from a threshold on a linear score
    score = 0.5 * X["f0"] + 0.3 * X["f3"] - 0.4 * X["f5"] + rs.normal(0, 0.3, n)
    y_clf = np.where(score > 0.3, 1, np.where(score < -0.3, -1, 0))
    full = X.assign(y_reg=y_reg, y_clf=y_clf)
    train = full.iloc[:400]
    val = full.iloc[400:]
    return train, val, list(X.columns)


def test_label_encoding_roundtrip() -> None:
    assert all(_XGB_TO_CLF[_CLF_TO_XGB[c]] == c for c in (-1, 0, 1))


def test_xgboost_regressor_fits_and_beats_baseline(synthetic) -> None:
    train, val, feats = synthetic
    m = XGBoostRegressor(feature_cols=feats, n_estimators=50, early_stopping_rounds=10).fit(
        train, train["y_reg"].values, X_val=val, y_val=val["y_reg"].values
    )
    pred = m.predict(val)
    # MSE much lower than predicting zero
    mse_model = float(np.mean((val["y_reg"].values - pred) ** 2))
    mse_zero = float(np.mean(val["y_reg"].values ** 2))
    assert mse_model < mse_zero * 0.7


def test_xgboost_classifier_predicts_ternary_labels(synthetic) -> None:
    train, val, feats = synthetic
    m = XGBoostClassifier(feature_cols=feats, n_estimators=50, early_stopping_rounds=10).fit(
        train, train["y_clf"].values, X_val=val, y_val=val["y_clf"].values
    )
    pred = m.predict(val)
    assert set(np.unique(pred)).issubset({-1, 0, 1})
    proba = m.predict_proba(val)
    assert proba.shape == (len(val), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_xgboost_save_load_roundtrip(synthetic) -> None:
    train, val, feats = synthetic
    with tempfile.TemporaryDirectory() as tmp:
        for ctor, target in [
            (XGBoostRegressor, "y_reg"),
            (XGBoostClassifier, "y_clf"),
        ]:
            m = ctor(feature_cols=feats, n_estimators=20, early_stopping_rounds=5).fit(
                train, train[target].values, X_val=val, y_val=val[target].values
            )
            before = m.predict(val)
            p = Path(tmp) / f"{ctor.__name__}"
            m.save(p)
            reloaded = ctor.load(p)
            after = reloaded.predict(val)
            np.testing.assert_array_equal(before, after)


def test_xgboost_handles_nan_natively(synthetic) -> None:
    """NaN in features must not crash XGBoost (native missing-value support)."""
    train, val, feats = synthetic
    train_with_nan = train.copy()
    train_with_nan.iloc[:50, train_with_nan.columns.get_loc("f0")] = np.nan
    m = XGBoostRegressor(feature_cols=feats, n_estimators=30, early_stopping_rounds=5).fit(
        train_with_nan, train_with_nan["y_reg"].values, X_val=val, y_val=val["y_reg"].values
    )
    assert m.booster_ is not None
    # Prediction with NaN in val should also work
    val_with_nan = val.copy()
    val_with_nan.iloc[0, val_with_nan.columns.get_loc("f0")] = np.nan
    pred = m.predict(val_with_nan)
    assert np.isfinite(pred).all()
