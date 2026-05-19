"""Unit tests for individual technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import technical as ti


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=500, freq="1h", tz="UTC")
    rs = np.random.RandomState(0)
    close = 2000 + np.cumsum(rs.normal(0, 1, 500))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + np.abs(rs.normal(1, 0.2, 500)),
            "low": close - np.abs(rs.normal(1, 0.2, 500)),
            "close": close,
            "volume": 0,
        },
        index=rng,
    )


def test_log_returns_basic(ohlcv: pd.DataFrame) -> None:
    r = ti.log_returns(ohlcv["close"], 1)
    assert r.iloc[0] != r.iloc[0]  # first is NaN
    assert np.isclose(
        r.iloc[1], np.log(ohlcv["close"].iloc[1] / ohlcv["close"].iloc[0])
    )


def test_rsi_bounded(ohlcv: pd.DataFrame) -> None:
    r = ti.rsi(ohlcv["close"], 14)
    valid = r.dropna()
    assert valid.between(0, 100).all()
    assert valid.shape[0] > 400


def test_macd_columns(ohlcv: pd.DataFrame) -> None:
    m = ti.macd(ohlcv["close"])
    assert list(m.columns) == ["macd_line", "macd_signal", "macd_hist"]
    # hist == line - signal by construction
    np.testing.assert_allclose(
        m["macd_hist"].dropna().values,
        (m["macd_line"] - m["macd_signal"]).dropna().values,
        rtol=1e-12,
    )


def test_bollinger_pctb_in_unit_interval_mostly(ohlcv: pd.DataFrame) -> None:
    bb = ti.bollinger(ohlcv["close"], 20, 2.0)
    pctb = bb["bb_pctb"].dropna()
    # On Gaussian-like returns, ~95% of points fall inside the bands
    # On a Gaussian random walk (not iid Gaussian), ~85% of points sit inside
    # the bands — strictly less than the textbook 95% for iid data because
    # the rolling mean drifts with the walk. We just want to confirm the
    # band has a sensible interpretation, not the exact theoretical rate.
    inside = pctb.between(0, 1).mean()
    assert inside > 0.85


def test_atr_positive(ohlcv: pd.DataFrame) -> None:
    a = ti.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
    assert (a.dropna() > 0).all()


def test_build_technical_features_shapes_and_shift(ohlcv: pd.DataFrame) -> None:
    feats = ti.build_technical_features(ohlcv, feature_lag=1)
    assert len(feats) == len(ohlcv)
    # Row 0 must be entirely NaN due to shift(1)
    assert feats.iloc[0].isna().all()
    # All columns named, no duplicates
    assert feats.columns.is_unique


def test_build_technical_no_lag_vs_lag1(ohlcv: pd.DataFrame) -> None:
    """Shifting by 1 must equal computing without shift then shifting."""
    feats_no_lag = ti.build_technical_features(ohlcv, feature_lag=0)
    feats_lag = ti.build_technical_features(ohlcv, feature_lag=1)
    pd.testing.assert_frame_equal(feats_lag, feats_no_lag.shift(1))
