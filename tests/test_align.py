"""Tests for the H1 alignment logic.

These tests are the first line of defence against temporal leakage:
they construct an external series with a known release lag and verify
that no H1 timestamp gets a value released in its own future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.align import align_to_h1, build_aligned_dataset


def _h1(start: str = "2024-01-08", days: int = 3) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=24 * days, freq="1h", tz="UTC")


def test_alignment_uses_release_date_not_value_date() -> None:
    """A value with value_date=Jan 1 and release_date=Jan 15 must NOT appear
    before Jan 15 in the aligned series."""
    h1 = _h1("2024-01-01", days=20)
    external = pd.DataFrame(
        {
            "value_date": pd.Timestamp("2024-01-01", tz="UTC"),
            "release_date": [pd.Timestamp("2024-01-15", tz="UTC")],
            "value": [42.0],
        }
    )
    aligned = align_to_h1(h1, external, source_name="cpi")
    # Before release date → NaN
    assert aligned.loc[: pd.Timestamp("2024-01-14 23:00", tz="UTC")].isna().all()
    # On / after release date → 42
    assert (aligned.loc[pd.Timestamp("2024-01-15 00:00", tz="UTC"):] == 42.0).all()


def test_alignment_rejects_missing_release_date() -> None:
    h1 = _h1()
    bad = pd.DataFrame({"value_date": [h1[0]], "value": [1.0]})
    with pytest.raises(ValueError, match="release_date"):
        align_to_h1(h1, bad, source_name="x")


def test_alignment_requires_tz_aware_index() -> None:
    naive = pd.date_range("2024-01-08", periods=10, freq="1h")
    df = pd.DataFrame(
        {"release_date": [naive[0]], "value": [1.0]}
    )
    with pytest.raises(ValueError, match="tz-aware"):
        align_to_h1(naive, df, source_name="x")


def test_no_future_leakage_random_stress() -> None:
    """Random stress test: for many H1 timestamps and many random external
    points, no value used at t should have release_date > t."""
    rs = np.random.RandomState(0)
    h1 = _h1("2024-01-08", days=14)
    n_obs = 50
    release_idx = np.sort(rs.choice(len(h1), size=n_obs, replace=False))
    release_dates = h1[release_idx]
    external = pd.DataFrame(
        {
            "value_date": release_dates - pd.Timedelta(days=3),
            "release_date": release_dates,
            "value": rs.normal(size=n_obs),
        }
    )
    aligned = align_to_h1(h1, external, source_name="x")

    for t, val in aligned.dropna().items():
        idx = external.index[external["value"] == val][0]
        rdate = external.loc[idx, "release_date"]
        assert rdate <= t, f"LEAKAGE at {t}: used release_date {rdate}"


def test_build_aligned_dataset_adds_columns() -> None:
    h1 = _h1("2024-01-08", days=5)
    ohlcv = pd.DataFrame(
        {
            "open": 2000.0, "high": 2001.0, "low": 1999.0,
            "close": 2000.5, "volume": 0,
        },
        index=h1,
    )
    ext_a = pd.DataFrame(
        {
            "value_date": [h1[0]],
            "release_date": [h1[0]],
            "value": [1.5],
        }
    )
    out = build_aligned_dataset(ohlcv, {"dxy": ext_a})
    assert "dxy" in out.columns
    assert (out["dxy"] == 1.5).all()
    assert out.shape == (len(h1), 6)
