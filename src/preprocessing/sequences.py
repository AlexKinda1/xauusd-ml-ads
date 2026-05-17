"""Convert a flat feature DataFrame into the two input shapes our models expect.

Two output formats:

  1. **Sequence** — sliding window of ``L`` consecutive bars, used by CNN /
     BiGRU / Chronos / FinCast. Shape ``[N, L, F]``.

  2. **Tabular** — one row per timestamp, used by XGBoost / Random Forest.
     Shape ``[N, F]`` (just the feature row at ``t``). Optionally augmented
     with rolling aggregates (``mean_L``, ``std_L``, etc.) of each feature
     over the past ``L`` bars.

Both formats drop:
- the first ``L - 1`` rows (insufficient history for a full window),
- rows where the target is NaN (typically the last ``h`` rows).

Anti-leakage: features are already shifted by ``feature_lag = 1`` in Phase 2,
so the value at index ``t`` describes bars ``<= t - 1``. Sequence windows
ending at ``t`` therefore use only past information by construction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sequence builder (DL models)
# ---------------------------------------------------------------------------


def build_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    lookback: int = 168,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return ``(X, y, end_timestamps)`` for the sequence DL models.

    Args:
        df: Feature dataframe with all features + target. Index must be
            chronological.
        feature_cols: Columns to use as inputs.
        target_col: Column to use as supervision signal.
        lookback: ``L`` — length of each input window.

    Returns:
        X : ndarray of shape ``[N, L, F]``
        y : ndarray of shape ``[N]``
        end_timestamps : DatetimeIndex of length ``N``; ``end_timestamps[i]``
            is the timestamp at which window ``i`` ends (i.e. the row at
            which the prediction is anchored).
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1.")
    feature_arr = df[list(feature_cols)].to_numpy(dtype="float64", copy=False)
    target_arr = df[target_col].to_numpy(dtype="float64", copy=False)
    index = df.index

    n_total = len(df)
    end_positions = np.arange(lookback - 1, n_total)
    valid = ~np.isnan(target_arr[end_positions])
    end_positions = end_positions[valid]

    if end_positions.size == 0:
        return (
            np.empty((0, lookback, len(feature_cols)), dtype="float64"),
            np.empty((0,), dtype="float64"),
            pd.DatetimeIndex([], tz=index.tz),
        )

    n_seq = end_positions.size
    n_feat = len(feature_cols)
    X = np.empty((n_seq, lookback, n_feat), dtype="float64")
    for i, end_pos in enumerate(end_positions):
        X[i] = feature_arr[end_pos - lookback + 1 : end_pos + 1]
    y = target_arr[end_positions]
    end_ts = index[end_positions]
    return X, y, end_ts


# ---------------------------------------------------------------------------
# Tabular builder (tree models)
# ---------------------------------------------------------------------------


def build_tabular(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    lookback: int = 168,
    aggregations: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for tree-based models.

    With ``aggregations = ()`` (default), each row in the output is simply
    the feature row at the corresponding timestamp — XGBoost and RF then see
    one observation per H1 bar.

    With aggregations like ``("mean", "std", "min", "max")``, every feature
    is augmented with rolling statistics computed over the past ``L`` bars,
    multiplying the column count by ``1 + len(aggregations)``.

    Args:
        df: Feature dataframe.
        feature_cols: Columns to use as inputs.
        target_col: Column to use as supervision signal.
        lookback: Window size for the rolling aggregations.
        aggregations: Names of pandas rolling-aggregations to apply
            (e.g. ``"mean"``, ``"std"``, ``"min"``, ``"max"``).

    Returns:
        X : DataFrame of shape ``[N, F * (1 + len(aggregations))]``
        y : Series of shape ``[N]``
    """
    X_now = df[list(feature_cols)].copy()
    pieces = [X_now]
    for agg in aggregations:
        rolling = X_now.rolling(lookback, min_periods=lookback).agg(agg)
        rolling.columns = [f"{c}_{agg}_{lookback}" for c in feature_cols]
        pieces.append(rolling)
    X = pd.concat(pieces, axis=1)

    # Drop the first lookback-1 rows where rolling aggregates are NaN
    # (only relevant if aggregations are requested).
    if aggregations:
        X = X.iloc[lookback - 1 :]
    y = df[target_col].loc[X.index]

    # Drop rows where target is NaN (tail of dataset).
    valid = y.notna()
    return X.loc[valid], y.loc[valid]
