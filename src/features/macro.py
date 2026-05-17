"""Transformations applied to aligned macro series.

The inputs here are already H1-aligned macro columns (produced by
``src.data.align``); they already respect release-date semantics. This module
only computes *transformations* — log-returns, deltas, ratios — which are
themselves causal as long as we apply the same global ``feature_lag`` shift
as for technical features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _log_return(s: pd.Series, lag: int) -> pd.Series:
    """``log(s_t / s_{t-lag})`` — safe to apply on positive-valued series."""
    return np.log(s.replace(0.0, np.nan)).diff(lag)


def _delta(s: pd.Series, lag: int) -> pd.Series:
    """``s_t - s_{t-lag}`` — used for level series like yields."""
    return s.diff(lag)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - mu) / sd


# Per-source transformation recipes. ``ratio`` series use deltas
# (level), ``price``-like series use log-returns.
TRANSFORMS: dict[str, str] = {
    "dxy":          "logret",
    "vix":          "logret",
    "spx500":       "logret",
    "brent_oil":    "logret",
    "us10y_yield":  "delta",
    "fed_funds":    "delta",
    "real_rate_10y": "delta",
    "cpi_yoy":      "delta",
}


def build_macro_features(
    aligned: pd.DataFrame,
    *,
    feature_lag: int = 1,
) -> pd.DataFrame:
    """Compute macro feature transformations from the H1-aligned dataset.

    Args:
        aligned: Output of :func:`src.data.align.build_aligned_dataset`. May
            contain OHLCV plus arbitrary macro columns. Macro columns not
            listed in :data:`TRANSFORMS` are still passed through as raw
            level (with the global shift applied).
        feature_lag: Same convention as :func:`src.features.technical.build_technical_features`.

    Returns:
        DataFrame with one column per macro-derived feature, indexed on
        ``aligned.index``. Empty if no macro columns are present.
    """
    # Exclude OHLCV and sentiment-prefixed columns (sentiment is handled by
    # src.features.sentiment to avoid feature duplication).
    from src.features.sentiment import SENTIMENT_PREFIXES
    excluded = {"open", "high", "low", "close", "volume"}
    macro_cols = [
        c for c in aligned.columns
        if c not in excluded and not c.startswith(SENTIMENT_PREFIXES)
    ]
    if not macro_cols:
        return pd.DataFrame(index=aligned.index)

    pieces: list[pd.Series] = []
    for col in macro_cols:
        s = aligned[col]
        kind = TRANSFORMS.get(col, "level")
        if kind == "logret":
            pieces.append(_log_return(s, 1).rename(f"{col}_logret_1"))
            pieces.append(_log_return(s, 24).rename(f"{col}_logret_24"))
            pieces.append(_zscore(s, 168).rename(f"{col}_z168"))
        elif kind == "delta":
            pieces.append(_delta(s, 1).rename(f"{col}_d1"))
            pieces.append(_delta(s, 24).rename(f"{col}_d24"))
            pieces.append(_zscore(s, 168).rename(f"{col}_z168"))
        else:  # "level" — pass-through + z-score
            pieces.append(s.rename(f"{col}_level"))
            pieces.append(_zscore(s, 168).rename(f"{col}_z168"))

    out = pd.concat(pieces, axis=1)
    if feature_lag > 0:
        out = out.shift(feature_lag)
    return out
