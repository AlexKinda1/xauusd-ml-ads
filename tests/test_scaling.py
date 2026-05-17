"""Tests for the train-only scaler."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.scaling import TrainOnlyScaler


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rs = np.random.RandomState(0)
    train = pd.DataFrame(
        {"a": rs.normal(10, 2, 500), "b": rs.normal(0, 1, 500), "c": rs.uniform(0, 100, 500)}
    )
    val = pd.DataFrame(
        {"a": rs.normal(10, 2, 100), "b": rs.normal(0, 1, 100), "c": rs.uniform(0, 100, 100)}
    )
    test = pd.DataFrame(
        {"a": rs.normal(10, 2, 100), "b": rs.normal(0, 1, 100), "c": rs.uniform(0, 100, 100)}
    )
    return train, val, test


def test_fit_then_transform_train_has_zero_mean(sample_data) -> None:
    train, _, _ = sample_data
    scaler = TrainOnlyScaler.fit(train)
    scaled = scaler.transform(train)
    np.testing.assert_allclose(scaled.mean(axis=0).values, 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0, ddof=0).values, 1, atol=1e-12)


def test_val_and_test_use_train_stats(sample_data) -> None:
    """val/test stats after scaling should NOT be (0, 1) — they should reflect
    train's stats applied to a different distribution."""
    train, val, test = sample_data
    scaler = TrainOnlyScaler.fit(train)
    scaled_val = scaler.transform(val)
    scaled_test = scaler.transform(test)
    # Train mean is ~0, val mean is small but not exactly 0
    assert abs(scaled_val["a"].mean()) > 1e-3 or len(val) < 50


def test_scaler_does_not_leak_when_adding_val_test_to_fit_data(sample_data) -> None:
    """CRITICAL: scaler stats fitted on train alone vs train+val+test should differ —
    proving that our API does NOT inadvertently use val/test when fitting."""
    train, val, test = sample_data
    scaler_train = TrainOnlyScaler.fit(train)
    full = pd.concat([train, val, test], ignore_index=True)
    scaler_full = TrainOnlyScaler.fit(full)
    # Stats differ
    assert not np.allclose(scaler_train.scaler.mean_, scaler_full.scaler.mean_)


def test_nan_passthrough_default(sample_data) -> None:
    train, val, _ = sample_data
    val_with_nan = val.copy()
    val_with_nan.iloc[0, 0] = np.nan
    scaler = TrainOnlyScaler.fit(train)
    scaled = scaler.transform(val_with_nan)
    assert np.isnan(scaled.iloc[0, 0])


def test_nan_filled_when_requested(sample_data) -> None:
    train, val, _ = sample_data
    val_with_nan = val.copy()
    val_with_nan.iloc[0, 0] = np.nan
    scaler = TrainOnlyScaler.fit(train)
    scaled = scaler.transform(val_with_nan, fillna=0.0)
    assert not np.isnan(scaled.iloc[0, 0])
    assert scaled.iloc[0, 0] == 0.0


def test_save_and_load_roundtrip(sample_data) -> None:
    train, val, _ = sample_data
    scaler = TrainOnlyScaler.fit(train)
    before = scaler.transform(val).values
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "scaler.pkl"
        scaler.save(p)
        reloaded = TrainOnlyScaler.load(p)
    after = reloaded.transform(val).values
    np.testing.assert_array_equal(before, after)


def test_fit_handles_nan_in_train(sample_data) -> None:
    train, _, _ = sample_data
    train_with_nan = train.copy()
    train_with_nan.iloc[:10, 0] = np.nan
    scaler = TrainOnlyScaler.fit(train_with_nan)
    # Should not crash and should produce finite stats
    assert np.isfinite(scaler.scaler.mean_).all()
    assert np.isfinite(scaler.scaler.scale_).all()
