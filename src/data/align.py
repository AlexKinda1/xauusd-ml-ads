"""Align macro & sentiment sources onto the H1 XAU/USD grid (no leakage).

The cardinal rule:

    For every timestamp ``t`` in the H1 grid, the value of any external
    series used at ``t`` MUST have a ``release_date <= t``.

Implementation: ``pd.merge_asof`` with ``direction='backward'`` on
``release_date``. This guarantees that we only consider observations whose
release date is **at or before** the H1 timestamp.

The function is deliberately defensive — any external frame without a
``release_date`` column is rejected to prevent accidental leakage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Single-source alignment
# ---------------------------------------------------------------------------


def align_to_h1(
    h1_index: pd.DatetimeIndex,
    external: pd.DataFrame,
    *,
    value_col: str = "value",
    release_col: str = "release_date",
    source_name: str = "external",
) -> pd.Series:
    """As-of-join an external series onto an H1 datetime index.

    Args:
        h1_index: Target tz-aware H1 :class:`DatetimeIndex` (typically the
            XAU/USD index).
        external: Frame containing at least ``release_col`` and ``value_col``.
        value_col: Name of the value column in ``external``.
        release_col: Name of the release-date column in ``external``.
        source_name: Used to name the returned Series.

    Returns:
        Series aligned on ``h1_index``, named ``source_name``, possibly with
        leading NaNs before the first available release_date.

    Raises:
        ValueError: If ``external`` lacks ``release_col`` (anti-leakage guard).
    """
    if release_col not in external.columns:
        raise ValueError(
            f"External frame for {source_name!r} has no {release_col!r} column. "
            "Refusing to align to avoid look-ahead bias."
        )
    if value_col not in external.columns:
        raise ValueError(f"External frame for {source_name!r} has no {value_col!r} column.")
    if h1_index.tz is None:
        raise ValueError("h1_index must be tz-aware.")

    ext = (
        external[[release_col, value_col]]
        .dropna(subset=[release_col])
        .copy()
    )
    ext[release_col] = pd.to_datetime(ext[release_col])
    if ext[release_col].dt.tz is None:
        ext[release_col] = ext[release_col].dt.tz_localize(h1_index.tz)
    else:
        ext[release_col] = ext[release_col].dt.tz_convert(h1_index.tz)
    # Harmonise datetime precision. The XAU/USD index from the CSV is
    # datetime64[us, UTC] while Parquet round-trips often return
    # datetime64[ms, UTC]; merge_asof refuses to join across precisions.
    ext[release_col] = ext[release_col].astype("datetime64[ns, UTC]")
    target_index = h1_index.astype("datetime64[ns, UTC]")

    ext = ext.sort_values(release_col)
    left = pd.DataFrame({"_t": target_index}).sort_values("_t")

    merged = pd.merge_asof(
        left,
        ext,
        left_on="_t",
        right_on=release_col,
        direction="backward",
        allow_exact_matches=True,
    )
    series = pd.Series(
        merged[value_col].to_numpy(),
        index=h1_index,
        name=source_name,
    )
    n_filled = int(series.notna().sum())
    logger.info(
        "Aligned %s to H1 grid: %d/%d non-null (%.1f%%)",
        source_name, n_filled, len(series), 100.0 * n_filled / max(len(series), 1),
    )
    return series


# ---------------------------------------------------------------------------
# Multi-source orchestration
# ---------------------------------------------------------------------------


def build_aligned_dataset(
    ohlcv: pd.DataFrame,
    external_sources: dict[str, pd.DataFrame],
    *,
    value_col: str = "value",
    release_col: str = "release_date",
) -> pd.DataFrame:
    """Build the unified dataset: OHLCV + every external source aligned on H1.

    Args:
        ohlcv: H1 OHLCV DataFrame with a tz-aware DatetimeIndex.
        external_sources: Mapping ``column_name -> dataframe`` for each
            macro/sentiment series. Each frame must expose ``release_col``
            and ``value_col``.

    Returns:
        DataFrame indexed on ``ohlcv.index`` with original OHLCV columns
        plus one column per external source.
    """
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv must be indexed by DatetimeIndex.")
    if ohlcv.index.tz is None:
        raise ValueError("ohlcv must have a tz-aware index.")

    out = ohlcv.copy()
    for name, df in external_sources.items():
        out[name] = align_to_h1(
            ohlcv.index, df,
            value_col=value_col,
            release_col=release_col,
            source_name=name,
        )
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_external_parquets(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Load every ``macro_*.parquet`` / ``sentiment_*.parquet`` in a directory.

    File ``macro_dxy.parquet`` becomes key ``dxy``.
    File ``sentiment_fear_greed.parquet`` becomes key ``fear_greed``.
    """
    d = Path(directory)
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(d.glob("*.parquet")):
        stem = p.stem
        if stem.startswith("macro_"):
            key = stem[len("macro_"):]
        elif stem.startswith("sentiment_"):
            key = stem[len("sentiment_"):]
        else:
            continue
        out[key] = pd.read_parquet(p)
    logger.info("Loaded %d external sources from %s", len(out), d)
    return out


def save_aligned(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Persist the aligned dataset to Parquet."""
    p = Path(out_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", compression="snappy")
    logger.info("Saved aligned dataset (%d rows, %d cols) -> %s", len(df), df.shape[1], p)
    return p
