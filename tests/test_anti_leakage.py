"""Anti-leakage tests (CRITICAL).

These tests are run with ``pytest -m leakage`` and MUST always pass. They
verify that the temporal alignment never lets a feature at time ``t`` use
information released after ``t``.

Phase 1 covers source-alignment leakage. Phase 2 will extend this file
with feature-level checks (technical indicators, rolling stats, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.align import align_to_h1


pytestmark = pytest.mark.leakage


def _random_h1_grid(n_hours: int = 24 * 30, seed: int = 0) -> pd.DatetimeIndex:
    rs = np.random.RandomState(seed)
    base = pd.Timestamp("2023-01-01", tz="UTC")
    offsets = np.sort(rs.choice(n_hours, size=n_hours, replace=False))
    return pd.DatetimeIndex([base + pd.Timedelta(hours=int(o)) for o in offsets])


@pytest.mark.parametrize("seed", [42, 123, 456, 789, 2024])
def test_alignment_never_uses_future_releases(seed: int) -> None:
    """For 5 random configurations of external observations, no aligned cell
    is allowed to carry a value with ``release_date`` strictly greater than
    its H1 timestamp."""
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

    # For each non-NaN entry, recompute the maximum allowed release_date and verify.
    for t, val in aligned.dropna().items():
        candidates = external.loc[external["release_date"] <= t, "release_date"]
        assert not candidates.empty, f"No releases available at {t} but got {val}"
        latest_allowed = candidates.max()
        # The aligned value must come from an observation released at or before t.
        matching = external[
            (external["value"] == val) & (external["release_date"] <= t)
        ]
        assert not matching.empty, f"LEAKAGE at {t}: value {val} has release_date > {t}"
        # And it must be the most recent such observation.
        assert matching["release_date"].max() == latest_allowed
