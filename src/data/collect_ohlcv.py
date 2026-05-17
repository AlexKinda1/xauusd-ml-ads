"""Load and validate the XAU/USD H1 OHLCV CSV.

The raw CSV is the project's source of truth. This module is responsible for:
  - parsing dates and standardising column names,
  - applying the configured timezone,
  - running integrity checks (duplicates, monotonic index, OHLC coherence, gaps),
  - persisting a clean Parquet snapshot in ``data/interim/``.

The timezone of the raw feed is unknown a priori and is therefore configurable
via ``config/data.yaml`` (``ohlcv.timezone``). Defaults to UTC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data import validate as v
from src.utils.config import PROJECT_ROOT, load_data_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
STANDARD_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_raw_ohlcv(csv_path: str | Path, timezone: str = "UTC") -> pd.DataFrame:
    """Read the raw CSV into a tz-aware OHLCV DataFrame.

    Args:
        csv_path: Path to the raw CSV (e.g. ``data/raw/XAUUSD_H1.csv``).
        timezone: IANA timezone to localise naive timestamps with
            (default ``UTC``). The source file's timezone must be known —
            mis-localising will silently shift the entire series.

    Returns:
        DataFrame with a tz-aware DatetimeIndex named ``datetime`` and
        lower-cased columns ``open, high, low, close, volume``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the CSV header does not match the expected schema.
    """
    p = Path(csv_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"OHLCV file not found: {p}")

    df = pd.read_csv(p)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="raise")
    df = df.rename(
        columns={
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df.set_index("datetime").sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    else:
        df.index = df.index.tz_convert(timezone)

    df = df[STANDARD_COLUMNS].astype({c: "float64" for c in STANDARD_COLUMNS})
    logger.info("Loaded %d rows from %s (tz=%s)", len(df), p, timezone)
    return df


def validate_ohlcv(df: pd.DataFrame) -> list[v.ValidationIssue]:
    """Run all OHLCV integrity checks and return collected issues."""
    issues: list[v.ValidationIssue] = []
    issues += v.check_monotonic_index(df)
    issues += v.check_duplicates(df.reset_index(), subset=["datetime"])
    issues += v.check_ohlc_coherence(df)
    issues += v.check_no_nans(df, cols=["open", "high", "low", "close"])
    issues += v.check_h1_gaps(df)

    # Volume == 0 is expected on XAUUSD OTC feeds. Inform rather than warn.
    if (df["volume"] == 0).all():
        issues.append(
            v.ValidationIssue(
                severity=v.Severity.INFO,
                code="VOLUME_ALL_ZERO",
                message="Volume is zero across the entire dataset (typical of OTC XAUUSD feeds).",
            )
        )
    return issues


def summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a small descriptive summary suitable for a Phase-1 report."""
    return {
        "n_rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "span_years": round((df.index.max() - df.index.min()).days / 365.25, 2),
        "n_unique_days": int(df.index.normalize().nunique()),
        "price_min": float(df["low"].min()),
        "price_max": float(df["high"].max()),
        "median_h1_return_pct": float(
            (df["close"].pct_change().abs().median()) * 100
        ),
    }


def save_interim(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Persist the cleaned DataFrame as Parquet (Snappy compression)."""
    p = Path(out_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", compression="snappy")
    logger.info("Saved %d rows to %s", len(df), p)
    return p


def run(strict: bool = False) -> pd.DataFrame:
    """End-to-end Phase-1 OHLCV pipeline driven by ``config/data.yaml``.

    Args:
        strict: If True, abort on any ERROR-severity issue. WARNINGs always
            log but never abort.

    Returns:
        The cleaned, validated OHLCV DataFrame.
    """
    cfg = load_data_config()["ohlcv"]
    df = load_raw_ohlcv(cfg["raw_path"], timezone=cfg.get("timezone", "UTC"))
    issues = validate_ohlcv(df)
    for it in issues:
        logger.log(
            {"info": 20, "warning": 30, "error": 40}[it.severity.value], "%s", it
        )
    if strict:
        v.raise_if_errors(issues)
    save_interim(df, cfg["parquet_path"])
    return df
