"""Generic data-integrity validators used by all collectors.

Each helper returns a list of :class:`ValidationIssue` (empty if OK). Callers
decide whether to log, raise, or quarantine the offending rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Severity(str, Enum):
    """Issue severity. ``ERROR`` should abort downstream processing."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single integrity issue found by a validator."""

    severity: Severity
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


# ---------------------------------------------------------------------------
# OHLC / time-series helpers
# ---------------------------------------------------------------------------


def check_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> list[ValidationIssue]:
    """Flag duplicated rows (or duplicated keys when ``subset`` is given)."""
    dupes = df.duplicated(subset=subset, keep=False)
    n = int(dupes.sum())
    if n == 0:
        return []
    return [
        ValidationIssue(
            severity=Severity.ERROR,
            code="DUPLICATES",
            message=f"{n} duplicated rows (subset={subset})",
            details={"n_duplicates": n},
        )
    ]


def check_monotonic_index(df: pd.DataFrame) -> list[ValidationIssue]:
    """Ensure the DatetimeIndex is strictly increasing."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return [
            ValidationIssue(
                severity=Severity.ERROR,
                code="INDEX_NOT_DATETIME",
                message=f"Index is {type(df.index).__name__}, expected DatetimeIndex",
            )
        ]
    if not df.index.is_monotonic_increasing:
        return [
            ValidationIssue(
                severity=Severity.ERROR,
                code="INDEX_NOT_MONOTONIC",
                message="DatetimeIndex is not strictly increasing",
            )
        ]
    return []


def check_ohlc_coherence(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> list[ValidationIssue]:
    """Verify ``low <= min(open, close) <= max(open, close) <= high`` for every row."""
    issues: list[ValidationIssue] = []
    o, h, ll, c = df[open_col], df[high_col], df[low_col], df[close_col]

    bad_high = (h < o) | (h < c) | (h < ll)
    bad_low = (ll > o) | (ll > c) | (ll > h)

    n_bad_high = int(bad_high.sum())
    n_bad_low = int(bad_low.sum())

    if n_bad_high > 0:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="OHLC_HIGH_INCONSISTENT",
                message=f"{n_bad_high} rows where high is not the max of (open, high, low, close)",
                details={"n_rows": n_bad_high},
            )
        )
    if n_bad_low > 0:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="OHLC_LOW_INCONSISTENT",
                message=f"{n_bad_low} rows where low is not the min of (open, high, low, close)",
                details={"n_rows": n_bad_low},
            )
        )
    return issues


def check_no_nans(df: pd.DataFrame, cols: list[str] | None = None) -> list[ValidationIssue]:
    """Report any NaN found in the given columns (defaults to all)."""
    target = df[cols] if cols else df
    n_nan = int(target.isna().sum().sum())
    if n_nan == 0:
        return []
    per_col = target.isna().sum().to_dict()
    return [
        ValidationIssue(
            severity=Severity.WARNING,
            code="NAN_VALUES",
            message=f"{n_nan} NaN values found",
            details={"per_column": {k: int(v) for k, v in per_col.items() if v}},
        )
    ]


def check_h1_gaps(
    df: pd.DataFrame,
    *,
    max_weekday_gap_hours: int = 2,
    max_weekend_gap_hours: int = 60,
) -> list[ValidationIssue]:
    """Detect unexpectedly large gaps in an H1 DatetimeIndex.

    Gaps over a weekend (Friday late → Sunday late) are normal for FX/Gold and
    are filtered out — we only flag gaps that happen mid-week.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return [
            ValidationIssue(
                severity=Severity.ERROR,
                code="INDEX_NOT_DATETIME",
                message="check_h1_gaps requires a DatetimeIndex",
            )
        ]

    deltas = df.index.to_series().diff().dt.total_seconds().div(3600).dropna()
    suspicious = deltas[deltas > max_weekday_gap_hours]

    weekday_gaps = []
    weekend_gaps = []
    for ts, gap_h in suspicious.items():
        prev_ts = ts - pd.Timedelta(hours=gap_h)
        is_weekend = prev_ts.weekday() == 4 and ts.weekday() in (0, 6)
        if is_weekend and gap_h <= max_weekend_gap_hours:
            weekend_gaps.append((ts, gap_h))
        else:
            weekday_gaps.append((ts, gap_h))

    issues: list[ValidationIssue] = []
    if weekday_gaps:
        issues.append(
            ValidationIssue(
                severity=Severity.WARNING,
                code="WEEKDAY_GAPS",
                message=f"{len(weekday_gaps)} unexpected intra-week gaps in H1 index",
                details={
                    "n_gaps": len(weekday_gaps),
                    "examples": [(str(ts), float(g)) for ts, g in weekday_gaps[:5]],
                },
            )
        )
    if weekend_gaps:
        issues.append(
            ValidationIssue(
                severity=Severity.INFO,
                code="WEEKEND_GAPS",
                message=f"{len(weekend_gaps)} normal weekend gaps detected",
                details={"n_gaps": len(weekend_gaps)},
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarise(issues: list[ValidationIssue]) -> dict[str, int]:
    """Aggregate a list of issues by severity."""
    summary = {s.value: 0 for s in Severity}
    for it in issues:
        summary[it.severity.value] += 1
    return summary


def raise_if_errors(issues: list[ValidationIssue]) -> None:
    """Raise :class:`ValueError` if any issue has ``ERROR`` severity."""
    errors = [it for it in issues if it.severity == Severity.ERROR]
    if errors:
        msg = "\n".join(str(e) for e in errors)
        raise ValueError(f"Validation failed with {len(errors)} error(s):\n{msg}")
