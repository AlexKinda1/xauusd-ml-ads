"""Sentiment feature transformations.

Lightweight on purpose: sentiment data is sparse and noisy, so we resist
the temptation to over-engineer derived features. Each sentiment column
gets:

  - the raw aligned value (already H1-aligned via :mod:`src.data.align`),
  - a 24-bar moving average to smooth high-frequency noise,
  - a 168-bar z-score.

Like everything else, the result is shifted by ``feature_lag`` to preserve
the project's anti-leakage invariant.
"""

from __future__ import annotations

import pandas as pd

SENTIMENT_PREFIXES = ("fear_greed", "google_trends")


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - mu) / sd


def build_sentiment_features(
    aligned: pd.DataFrame,
    *,
    feature_lag: int = 1,
) -> pd.DataFrame:
    """Build sentiment-derived features from the aligned H1 dataset.

    Sentiment columns are identified by name prefixes
    (``fear_greed``, ``google_trends``). Other columns are ignored.
    """
    cols = [c for c in aligned.columns if c.startswith(SENTIMENT_PREFIXES)]
    if not cols:
        return pd.DataFrame(index=aligned.index)

    pieces: list[pd.Series] = []
    for col in cols:
        s = aligned[col]
        pieces.append(s.rename(f"{col}_level"))
        pieces.append(s.rolling(24, min_periods=24).mean().rename(f"{col}_ma24"))
        pieces.append(_zscore(s, 168).rename(f"{col}_z168"))

    out = pd.concat(pieces, axis=1)
    if feature_lag > 0:
        out = out.shift(feature_lag)
    return out
