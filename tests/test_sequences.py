"""Tests for the sequence and tabular builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sequences import build_sequences, build_tabular


@pytest.fixture
def feats_and_target() -> pd.DataFrame:
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "f1": rs.normal(0, 1, n),
            "f2": rs.normal(0, 1, n),
            "f3": rs.normal(0, 1, n),
            "y": rs.normal(0, 1, n),
        },
        index=idx,
    )
    # Last 24 rows have NaN target (simulate forward-looking target).
    df.iloc[-24:, df.columns.get_loc("y")] = np.nan
    return df


def test_build_sequences_shapes(feats_and_target: pd.DataFrame) -> None:
    X, y, end_ts = build_sequences(
        feats_and_target, ["f1", "f2", "f3"], "y", lookback=168
    )
    n_total = len(feats_and_target)
    # Sequences start at row 167 (0-indexed), then last 24 have NaN target.
    expected_n = n_total - 168 + 1 - 24
    assert X.shape == (expected_n, 168, 3)
    assert y.shape == (expected_n,)
    assert len(end_ts) == expected_n


def test_build_sequences_last_window_matches_last_valid_row(feats_and_target: pd.DataFrame) -> None:
    X, y, end_ts = build_sequences(feats_and_target, ["f1", "f2", "f3"], "y", lookback=168)
    # The very last window ends at the last row with a valid target.
    last_valid_t = feats_and_target.dropna(subset=["y"]).index[-1]
    assert end_ts[-1] == last_valid_t
    # And its content equals the corresponding slice.
    pos = feats_and_target.index.get_loc(last_valid_t)
    expected = feats_and_target[["f1", "f2", "f3"]].iloc[pos - 167 : pos + 1].to_numpy()
    np.testing.assert_array_equal(X[-1], expected)


def test_build_sequences_uses_only_past_within_window(feats_and_target: pd.DataFrame) -> None:
    """Window ending at t must use rows [t-L+1, t]."""
    X, _, end_ts = build_sequences(feats_and_target, ["f1", "f2", "f3"], "y", lookback=24)
    sample_t = end_ts[100]
    pos = feats_and_target.index.get_loc(sample_t)
    expected = feats_and_target[["f1", "f2", "f3"]].iloc[pos - 23 : pos + 1].to_numpy()
    np.testing.assert_array_equal(X[100], expected)


def test_build_tabular_default_no_aggregations(feats_and_target: pd.DataFrame) -> None:
    X, y = build_tabular(feats_and_target, ["f1", "f2", "f3"], "y", lookback=168)
    # Without aggregations we keep every row with a valid target.
    expected_n = len(feats_and_target) - 24
    assert len(X) == expected_n
    assert list(X.columns) == ["f1", "f2", "f3"]
    assert len(y) == expected_n


def test_build_tabular_with_aggregations(feats_and_target: pd.DataFrame) -> None:
    X, y = build_tabular(
        feats_and_target, ["f1", "f2", "f3"], "y",
        lookback=24, aggregations=("mean", "std"),
    )
    # Columns: 3 raw + 3 mean + 3 std = 9
    assert X.shape[1] == 9
    assert any(c.endswith("_mean_24") for c in X.columns)
    assert any(c.endswith("_std_24") for c in X.columns)
    # No NaN in the aggregate columns (lookback-1 warm-up rows are dropped)
    assert not X.isna().any().any()
