"""Chronological train / val / test splits with embargo + walk-forward folds.

The single anti-leakage rule for splits:

    train_end + embargo  <=  val_start
    val_end   + embargo  <=  test_start

Where ``embargo = h = horizon`` (default 24 bars). Practically, the last
``embargo`` rows of each segment are *dropped* — the target at those rows
would depend on prices that already belong to the following segment.

Random shuffles are explicitly forbidden. All splits respect strict
chronological order. Splits are computed from the row count, not from
calendar dates, so re-running on the same dataset is bit-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class Split:
    """A contiguous slice of a DatetimeIndex with named role."""

    name: str
    start_idx: int        # inclusive
    end_idx: int          # exclusive
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp

    @property
    def n_rows(self) -> int:
        return self.end_idx - self.start_idx

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Split({self.name!r}, rows={self.n_rows}, "
            f"{self.start_ts} -> {self.end_ts})"
        )


# ---------------------------------------------------------------------------
# Single 70/15/15 split with embargo
# ---------------------------------------------------------------------------


def chronological_split(
    index: pd.DatetimeIndex,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_bars: int = 24,
) -> dict[str, Split]:
    """Compute strict chronological train/val/test splits.

    Args:
        index: The DatetimeIndex of the feature dataset (already in
            chronological order).
        train_ratio, val_ratio, test_ratio: Must sum to <= 1.0.
        embargo_bars: Number of bars dropped at the END of each segment
            so that the target at those rows does not leak into the next
            segment. Use ``h`` (the prediction horizon).

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` -> :class:`Split`.

    Raises:
        ValueError: If ratios are inconsistent or the index is too short.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("chronological_split requires a DatetimeIndex.")
    if not index.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing (no shuffling).")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6 and total > 1.0:
        raise ValueError(f"Ratios sum to {total:.4f}; must be <= 1.0.")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0.")

    n = len(index)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    test_end = val_end + int(n * test_ratio)

    train_keep_end = train_end - embargo_bars
    val_start = train_end
    val_keep_end = val_end - embargo_bars
    test_start = val_end
    test_keep_end = test_end

    if min(train_keep_end - 0, val_keep_end - val_start, test_keep_end - test_start) <= 0:
        raise ValueError(
            f"Index too short for embargo={embargo_bars}: "
            f"train={train_keep_end}, val={val_keep_end - val_start}, "
            f"test={test_keep_end - test_start}"
        )

    return {
        "train": Split("train", 0, train_keep_end, index[0], index[train_keep_end - 1]),
        "val": Split("val", val_start, val_keep_end, index[val_start], index[val_keep_end - 1]),
        "test": Split("test", test_start, test_keep_end, index[test_start], index[test_keep_end - 1]),
    }


# ---------------------------------------------------------------------------
# Walk-forward expanding validation
# ---------------------------------------------------------------------------


def walk_forward_expanding(
    index: pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    embargo_bars: int = 24,
    initial_train_ratio: float = 0.40,
) -> Iterator[tuple[Split, Split]]:
    """Generate walk-forward expanding (train, val) pairs.

    Fold ``k`` (0-indexed) has:

        train : [0, t_k)
        val   : [t_k + embargo, t_k + embargo + fold_size)

    Where ``t_0 = initial_train_ratio * n`` and each subsequent fold
    extends the training window by one fold-size.

    Args:
        index: The full DatetimeIndex.
        n_folds: Number of (train, val) pairs to yield.
        embargo_bars: Bars dropped at the train/val boundary.
        initial_train_ratio: How much of the data is in the first fold's
            training window. The remaining ``1 - initial_train_ratio`` is
            divided into ``n_folds`` equal validation windows.

    Yields:
        Tuples ``(train_split, val_split)``.

    Raises:
        ValueError: If the index is too short for the requested folds.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")
    if not (0.0 < initial_train_ratio < 1.0):
        raise ValueError("initial_train_ratio must be in (0, 1).")

    n = len(index)
    initial_train = int(n * initial_train_ratio)
    remaining = n - initial_train
    fold_size = remaining // n_folds
    if fold_size <= embargo_bars:
        raise ValueError(
            f"Index too short: fold_size={fold_size} <= embargo={embargo_bars}"
        )

    for k in range(n_folds):
        train_end = initial_train + k * fold_size
        train_keep_end = train_end - embargo_bars
        val_start = train_end
        val_end = train_end + fold_size
        val_keep_end = val_end - embargo_bars

        if val_keep_end > n or train_keep_end <= 0:
            break

        yield (
            Split("train", 0, train_keep_end, index[0], index[train_keep_end - 1]),
            Split(
                f"val_fold{k}", val_start, val_keep_end,
                index[val_start], index[val_keep_end - 1],
            ),
        )


# ---------------------------------------------------------------------------
# Helpers for downstream code
# ---------------------------------------------------------------------------


def apply_split(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Return the rows of ``df`` belonging to a :class:`Split`."""
    return df.iloc[split.start_idx : split.end_idx]
