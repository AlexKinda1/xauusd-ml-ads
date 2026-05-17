"""Tests for chronological splits and walk-forward folds."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.splits import (
    apply_split,
    chronological_split,
    walk_forward_expanding,
)


@pytest.fixture
def h1_index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=10_000, freq="1h", tz="UTC")


def test_chronological_split_ratios(h1_index: pd.DatetimeIndex) -> None:
    splits = chronological_split(h1_index, embargo_bars=24)
    n = len(h1_index)
    # Ratios approximately respected (within rounding + embargo)
    assert abs(splits["train"].n_rows / n - 0.70) < 0.01
    assert abs(splits["val"].n_rows / n - 0.15) < 0.01
    assert abs(splits["test"].n_rows / n - 0.15) < 0.01


def test_chronological_split_no_overlap(h1_index: pd.DatetimeIndex) -> None:
    s = chronological_split(h1_index, embargo_bars=24)
    assert s["train"].end_idx <= s["val"].start_idx
    assert s["val"].end_idx <= s["test"].start_idx


def test_chronological_split_embargo(h1_index: pd.DatetimeIndex) -> None:
    s = chronological_split(h1_index, embargo_bars=24)
    # Gap >= 24 between train end and val start
    assert s["val"].start_idx - s["train"].end_idx >= 24
    # Gap >= 24 between val end and test start
    assert s["test"].start_idx - s["val"].end_idx >= 24


def test_chronological_split_strict_monotonic(h1_index: pd.DatetimeIndex) -> None:
    s = chronological_split(h1_index, embargo_bars=24)
    assert s["train"].end_ts < s["val"].start_ts
    assert s["val"].end_ts < s["test"].start_ts


def test_chronological_split_rejects_shuffled() -> None:
    rng = pd.date_range("2020-01-01", periods=100, freq="1h", tz="UTC")
    shuffled = rng[[5, 2, 8, 1, 0] + list(range(10, 100))]
    with pytest.raises(ValueError, match="monotonically"):
        chronological_split(shuffled)


def test_chronological_split_too_short() -> None:
    rng = pd.date_range("2020-01-01", periods=50, freq="1h", tz="UTC")
    with pytest.raises(ValueError):
        chronological_split(rng, embargo_bars=24)


def test_walk_forward_expanding_folds(h1_index: pd.DatetimeIndex) -> None:
    folds = list(walk_forward_expanding(h1_index, n_folds=5, embargo_bars=24))
    assert len(folds) == 5
    # Each train is strictly larger than the previous
    for i in range(1, len(folds)):
        assert folds[i][0].n_rows > folds[i - 1][0].n_rows
    # Each val window is strictly after the corresponding train end
    for train, val in folds:
        assert train.end_ts < val.start_ts


def test_walk_forward_no_overlap_with_embargo(h1_index: pd.DatetimeIndex) -> None:
    for train, val in walk_forward_expanding(h1_index, n_folds=5, embargo_bars=24):
        gap = val.start_idx - train.end_idx
        assert gap >= 24, f"Gap {gap} < embargo at fold {val.name}"


def test_apply_split_returns_correct_rows(h1_index: pd.DatetimeIndex) -> None:
    df = pd.DataFrame({"x": range(len(h1_index))}, index=h1_index)
    splits = chronological_split(h1_index, embargo_bars=24)
    train_df = apply_split(df, splits["train"])
    assert len(train_df) == splits["train"].n_rows
    assert train_df.index[0] == splits["train"].start_ts
    assert train_df.index[-1] == splits["train"].end_ts
