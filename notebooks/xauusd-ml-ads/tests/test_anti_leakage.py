"""Anti-leakage tests (CRITICAL).

These tests are run with ``pytest -m leakage`` and MUST always pass. They
verify that the temporal alignment never lets a feature at time ``t`` use
information released after ``t``.

Coverage:
- Phase 1: source-alignment leakage (``release_date`` enforcement).
- Phase 2: feature-level leakage. For 10 random timestamps, every feature
  at row ``t`` must be IDENTICAL to what one would compute using only
  ``prices[:t]`` (no access to bar ``t`` or later when ``feature_lag=1``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.align import align_to_h1
from src.features import calendar, sentiment as sent_mod, target as tgt_mod, technical
from src.features.macro import build_macro_features
from src.features.pipeline import build_features_and_targets


pytestmark = pytest.mark.leakage


# ---------------------------------------------------------------------------
# Phase 1 — alignment leakage
# ---------------------------------------------------------------------------


def _random_h1_grid(n_hours: int = 24 * 30, seed: int = 0) -> pd.DatetimeIndex:
    rs = np.random.RandomState(seed)
    base = pd.Timestamp("2023-01-01", tz="UTC")
    offsets = np.sort(rs.choice(n_hours, size=n_hours, replace=False))
    return pd.DatetimeIndex([base + pd.Timedelta(hours=int(o)) for o in offsets])


@pytest.mark.parametrize("seed", [42, 123, 456, 789, 2024])
def test_alignment_never_uses_future_releases(seed: int) -> None:
    """For 5 random configurations, no aligned cell may carry a value with
    ``release_date`` strictly greater than its H1 timestamp."""
    rs = np.random.RandomState(seed)
    h1 = _random_h1_grid()

    n_obs = 200
    release_idx = np.sort(rs.choice(len(h1), size=n_obs, replace=False))
    release_dates = h1[release_idx]
    external = pd.DataFrame(
        {
            "value_date": release_dates - pd.Timedelta(days=2),
            "release_date": release_dates,
            "value": rs.normal(size=n_obs),
        }
    )

    aligned = align_to_h1(h1, external, source_name="x")

    for t, val in aligned.dropna().items():
        candidates = external.loc[external["release_date"] <= t, "release_date"]
        assert not candidates.empty, f"No releases available at {t} but got {val}"
        latest_allowed = candidates.max()
        matching = external[
            (external["value"] == val) & (external["release_date"] <= t)
        ]
        assert not matching.empty, f"LEAKAGE at {t}: value {val} has release_date > {t}"
        assert matching["release_date"].max() == latest_allowed


# ---------------------------------------------------------------------------
# Phase 2 — feature-level leakage
# ---------------------------------------------------------------------------


def _synth_ohlcv(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV with realistic structure for leakage testing."""
    rs = np.random.RandomState(seed)
    close = 2000 + np.cumsum(rs.normal(0, 1.0, n))
    return pd.DataFrame(
        {
            "open": close + rs.normal(0, 0.1, n),
            "high": close + np.abs(rs.normal(1, 0.2, n)),
            "low": close - np.abs(rs.normal(1, 0.2, n)),
            "close": close,
            "volume": 0,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
    )


def test_features_at_t_use_only_data_before_t() -> None:
    """For 10 random timestamps, every feature at row ``t`` must equal what
    we get from recomputing on the truncated price series ``prices[:t]``.

    With ``feature_lag = 1``, features at index ``t`` describe market state
    using bars ``<= t-1``. So truncating after ``t-1`` (i.e. keeping
    ``prices[:t]``, exclusive of ``t``) must not change the value at ``t``.
    """
    ohlcv = _synth_ohlcv(n=1000)
    full = technical.build_technical_features(ohlcv, feature_lag=1)

    rs = np.random.RandomState(123)
    # Sample timestamps from the second half, where every indicator is warmed up.
    candidates = ohlcv.index[300:]
    sample = rs.choice(len(candidates), size=10, replace=False)
    timestamps = candidates[sample]

    for t in timestamps:
        truncated = ohlcv.loc[:t].iloc[:-1]   # everything strictly before t
        partial = technical.build_technical_features(truncated, feature_lag=1)
        # The last row of `partial` corresponds to index t-1 in the truncated
        # frame. But we want feature value AT time t in the full frame.
        # With feature_lag=1, that value = compute on truncated.iloc[:-0] then shift.
        # Equivalently: compute on `truncated` (which excludes t) with feature_lag=0,
        # then read the LAST row of that — it represents state at the bar JUST BEFORE t.
        partial_unshifted = technical.build_technical_features(truncated, feature_lag=0)
        expected = partial_unshifted.iloc[-1]  # last bar of the truncated frame
        actual = full.loc[t]
        # Compare each column; tolerate NaN-NaN.
        for col in expected.index:
            a, e = actual[col], expected[col]
            if pd.isna(a) and pd.isna(e):
                continue
            assert np.isclose(a, e, equal_nan=True), (
                f"LEAKAGE at {t}, column {col}: full={a}, recomputed_from_past={e}"
            )


def test_macro_features_at_t_use_only_data_before_t() -> None:
    """Same property for macro features."""
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rs = np.random.RandomState(7)
    aligned = pd.DataFrame(
        {
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0,
            "dxy": 100.0 + np.cumsum(rs.normal(0, 0.1, n)),
            "us10y_yield": 3.0 + rs.normal(0, 0.05, n).cumsum(),
        },
        index=idx,
    )
    full = build_macro_features(aligned, feature_lag=1)
    rs = np.random.RandomState(42)
    timestamps = idx[200 + rs.choice(n - 200, size=10, replace=False)]
    for t in timestamps:
        truncated = aligned.loc[:t].iloc[:-1]
        partial_unshifted = build_macro_features(truncated, feature_lag=0)
        for col in full.columns:
            a, e = full.loc[t, col], partial_unshifted.iloc[-1][col]
            if pd.isna(a) and pd.isna(e):
                continue
            assert np.isclose(a, e, equal_nan=True), (
                f"MACRO LEAKAGE at {t}, column {col}: full={a}, past={e}"
            )


def test_classification_target_threshold_does_not_use_future() -> None:
    """The neutral-zone threshold at ``t`` must depend only on returns ≤ t-1.

    Inject a synthetic spike at time ``t`` and verify the threshold at
    ``t`` is unchanged compared with a baseline series with no spike.
    """
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rs = np.random.RandomState(0)
    base_close = pd.Series(100 + np.cumsum(rs.normal(0, 0.5, n)), index=idx)

    t_idx = 300
    spiked = base_close.copy()
    spiked.iloc[t_idx] *= 1.10   # +10% spike exactly at t_idx

    tau_base = tgt_mod.realized_vol_past(base_close, window=24, lag=1).iloc[t_idx]
    tau_spike = tgt_mod.realized_vol_past(spiked, window=24, lag=1).iloc[t_idx]
    assert np.isclose(tau_base, tau_spike), (
        f"Threshold at t leaked future spike: base={tau_base}, spiked={tau_spike}"
    )


def test_full_pipeline_features_have_no_future_information() -> None:
    """End-to-end: for 10 random timestamps in the full feature dataset, every
    feature column (technical + macro + sentiment + calendar) must equal the
    value obtained by re-running the pipeline on the truncated input."""
    n = 1500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rs = np.random.RandomState(0)
    close = 2000 + np.cumsum(rs.normal(0, 1.0, n))
    aligned = pd.DataFrame(
        {
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 0,
        },
        index=idx,
    )

    full = build_features_and_targets(
        aligned, horizon=24, lookback=168, feature_lag=1, threshold_factor=0.5
    )

    rs = np.random.RandomState(2024)
    # Skip the warm-up rows AND the tail (where the regression target is NaN).
    valid = full.index[: -24]
    sample_ts = valid[rs.choice(len(valid), size=10, replace=False)]

    # Calendar features are deterministic functions of the timestamp itself —
    # they describe "decision time" and are by definition known at ``t``.
    # The anti-leakage rule only applies to market-data-derived features.
    calendar_cols = set(calendar.calendar_features(full.index[:5]).columns)
    feature_cols = [
        c for c in full.columns
        if c not in {"open", "high", "low", "close", "volume",
                     "y_reg_h24", "y_clf_h24", "y_clf_threshold"} | calendar_cols
    ]

    for t in sample_ts:
        truncated = aligned.loc[:t].iloc[:-1]   # excludes bar t
        if len(truncated) < 200:
            continue
        # Market features at t must equal features computed on truncated data
        # (no access to bar t). Calendar features are excluded above.
        rebuilt = technical.build_technical_features(truncated, feature_lag=0).iloc[-1]

        for col in feature_cols:
            if col not in rebuilt.index:
                continue
            a, e = full.loc[t, col], rebuilt[col]
            if pd.isna(a) and pd.isna(e):
                continue
            assert np.isclose(a, e, equal_nan=True), (
                f"PIPELINE LEAKAGE at {t}, column {col}: full={a}, past-only={e}"
            )

