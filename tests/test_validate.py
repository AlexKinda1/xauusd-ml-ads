"""Unit tests for the generic validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import validate as v


def _make_ohlcv(n: int = 24, freq: str = "1h") -> pd.DataFrame:
    rng = pd.date_range("2024-01-08 00:00", periods=n, freq=freq, tz="UTC")
    rs = np.random.RandomState(0)
    close = 2000 + np.cumsum(rs.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close + rs.normal(0, 0.1, n),
            "high": close + np.abs(rs.normal(1, 0.2, n)),
            "low": close - np.abs(rs.normal(1, 0.2, n)),
            "close": close,
            "volume": 0,
        },
        index=rng,
    ).rename_axis("datetime")


def test_check_monotonic_index_ok() -> None:
    assert v.check_monotonic_index(_make_ohlcv()) == []


def test_check_monotonic_index_detects_unsorted() -> None:
    df = _make_ohlcv()
    df = pd.concat([df.iloc[-1:], df.iloc[:-1]])
    issues = v.check_monotonic_index(df)
    assert len(issues) == 1 and issues[0].severity == v.Severity.ERROR


def test_check_ohlc_coherence_ok() -> None:
    df = _make_ohlcv()
    assert v.check_ohlc_coherence(df) == []


def test_check_ohlc_coherence_detects_bad_high() -> None:
    df = _make_ohlcv()
    df.loc[df.index[0], "high"] = df.loc[df.index[0], "low"] - 1   # high < low
    issues = v.check_ohlc_coherence(df)
    assert any(it.code == "OHLC_HIGH_INCONSISTENT" for it in issues)


def test_check_duplicates_detects() -> None:
    df = _make_ohlcv().reset_index()
    df = pd.concat([df, df.iloc[:1]])
    issues = v.check_duplicates(df, subset=["datetime"])
    assert len(issues) == 1
    assert issues[0].details["n_duplicates"] >= 2


def test_check_h1_gaps_classifies_weekend_correctly() -> None:
    # Build an index with Friday 22:00 UTC → Sunday 22:00 UTC = 48h weekend gap
    idx = pd.DatetimeIndex(
        [
            "2024-03-15 22:00",   # Friday
            "2024-03-17 22:00",   # Sunday
            "2024-03-17 23:00",
        ],
        tz="UTC",
    )
    df = pd.DataFrame({"x": 1.0}, index=idx)
    issues = v.check_h1_gaps(df)
    codes = {i.code for i in issues}
    assert "WEEKEND_GAPS" in codes
    assert "WEEKDAY_GAPS" not in codes


def test_check_h1_gaps_detects_weekday_gap() -> None:
    idx = pd.DatetimeIndex(
        ["2024-03-19 10:00", "2024-03-19 18:00"],   # Tuesday 8h gap
        tz="UTC",
    )
    df = pd.DataFrame({"x": 1.0}, index=idx)
    issues = v.check_h1_gaps(df)
    assert any(i.code == "WEEKDAY_GAPS" for i in issues)


def test_raise_if_errors() -> None:
    issues = [v.ValidationIssue(v.Severity.ERROR, "X", "boom")]
    with pytest.raises(ValueError):
        v.raise_if_errors(issues)
    assert v.raise_if_errors([v.ValidationIssue(v.Severity.WARNING, "X", "ok")]) is None
