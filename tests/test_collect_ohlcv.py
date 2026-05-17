"""Tests for the OHLCV loader and validator using the real project CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import collect_ohlcv

CSV_PATH = Path("XAUUSD_H1.csv")


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / CSV_PATH).exists(),
    reason="XAUUSD_H1.csv not present (project data not yet committed)",
)


def test_load_returns_tz_aware_index(project_root: Path) -> None:
    df = collect_ohlcv.load_raw_ohlcv(project_root / CSV_PATH, timezone="UTC")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


def test_load_has_expected_columns(project_root: Path) -> None:
    df = collect_ohlcv.load_raw_ohlcv(project_root / CSV_PATH)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_load_is_sorted(project_root: Path) -> None:
    df = collect_ohlcv.load_raw_ohlcv(project_root / CSV_PATH)
    assert df.index.is_monotonic_increasing


def test_no_critical_validation_errors(project_root: Path) -> None:
    """The real CSV must have a sane structure (no duplicates, coherent OHLC)."""
    df = collect_ohlcv.load_raw_ohlcv(project_root / CSV_PATH)
    issues = collect_ohlcv.validate_ohlcv(df)
    errors = [i for i in issues if i.severity.value == "error"]
    assert not errors, f"Unexpected ERROR-level issues: {errors}"


def test_summary_stats_runs(project_root: Path) -> None:
    df = collect_ohlcv.load_raw_ohlcv(project_root / CSV_PATH)
    stats = collect_ohlcv.summary_stats(df)
    assert stats["n_rows"] > 50_000
    assert stats["span_years"] > 15
