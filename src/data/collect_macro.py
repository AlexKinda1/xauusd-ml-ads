"""Download macro time-series and persist them with explicit release dates.

Anti-leakage rule of thumb: every macro observation has TWO timestamps —

  - ``value_date``  : the date the observation describes (e.g. CPI for Jan 2024).
  - ``release_date``: the date that observation became publicly available
                      (CPI for Jan 2024 is published mid-Feb 2024).

Downstream alignment (:mod:`src.data.align`) joins macro series on
``release_date``, never ``value_date``.

Network access required. In sandboxed environments without outbound HTTP,
use ``run(skip_missing_key=True)`` and the module will gracefully skip
unreachable sources, returning whatever it managed to fetch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_data_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

Provider = Literal["fred", "yfinance"]


@dataclass(frozen=True)
class MacroSeriesConfig:
    """Configuration for a single macro series."""

    name: str
    provider: Provider
    identifier: str
    frequency: str           # "daily" | "monthly" | "weekly"
    release_lag_days: int    # business days between value_date and release_date


# Default release-lag heuristics if not overridden in YAML.
DEFAULT_LAGS: dict[str, int] = {
    "daily": 1,        # FRED daily series published next business day
    "weekly": 5,       # weekly series usually released 5 days after week-end
    "monthly": 30,     # conservative; CPI overridden to 14 in YAML
}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _fetch_fred(series_id: str, api_key: str | None = None) -> pd.Series:
    """Fetch a FRED series. Requires ``FRED_API_KEY`` env var or arg."""
    try:
        from fredapi import Fred  # type: ignore
    except ImportError as e:
        raise RuntimeError("fredapi not installed; run `poetry install`") from e

    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY env var not set; copy .env.example to .env")

    fred = Fred(api_key=key)
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s.dropna()


def _fetch_yfinance(ticker: str) -> pd.Series:
    """Fetch the daily Close of a yfinance ticker over the full available history.

    Recent yfinance releases default to ``period="1mo"`` when neither ``period``
    nor ``start``/``end`` are passed, which would silently truncate every series
    to the last month. We force ``period="max"`` to get the full history.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as e:
        raise RuntimeError("yfinance not installed; run `poetry install`") from e

    data = yf.download(
        ticker,
        period="max",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    # Multi-ticker DataFrames have MultiIndex columns; reduce.
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    close.name = ticker
    return close.dropna()


# ---------------------------------------------------------------------------
# Release-date computation
# ---------------------------------------------------------------------------


def add_release_dates(
    values: pd.Series,
    release_lag_days: int,
) -> pd.DataFrame:
    """Convert a single-indexed Series into a ``[value_date, release_date, value]`` frame.

    ``release_date = value_date + release_lag_days`` (calendar days; conservative).
    """
    df = values.rename("value").to_frame()
    df.index.name = "value_date"
    df = df.reset_index()
    df["release_date"] = df["value_date"] + pd.Timedelta(days=release_lag_days)
    return df[["value_date", "release_date", "value"]]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _series_configs() -> list[MacroSeriesConfig]:
    """Materialise series configurations from ``config/data.yaml``."""
    raw = load_data_config()["macro"]["sources"]
    configs: list[MacroSeriesConfig] = []
    for name, spec in raw.items():
        provider = spec["provider"]
        identifier = spec["ticker"] if provider == "yfinance" else spec["series_id"]
        freq = spec["frequency"]
        lag = spec.get("release_lag_days", DEFAULT_LAGS[freq])
        configs.append(
            MacroSeriesConfig(
                name=name,
                provider=provider,
                identifier=identifier,
                frequency=freq,
                release_lag_days=lag,
            )
        )
    return configs


def fetch_series(cfg: MacroSeriesConfig) -> pd.DataFrame:
    """Fetch one series and return it with explicit release dates."""
    logger.info("Fetching %s (%s:%s)", cfg.name, cfg.provider, cfg.identifier)
    if cfg.provider == "fred":
        raw = _fetch_fred(cfg.identifier)
    elif cfg.provider == "yfinance":
        raw = _fetch_yfinance(cfg.identifier)
    else:  # pragma: no cover - guarded by Literal type
        raise ValueError(f"Unknown provider: {cfg.provider}")
    return add_release_dates(raw, release_lag_days=cfg.release_lag_days)


def save_series(df: pd.DataFrame, name: str, out_dir: Path | str = "data/external") -> Path:
    """Persist a macro series Parquet to ``data/external/macro_<name>.parquet``."""
    d = Path(out_dir)
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"macro_{name}.parquet"
    df.to_parquet(out, engine="pyarrow", compression="snappy")
    logger.info("Saved %s (%d rows) -> %s", name, len(df), out)
    return out


def run(skip_on_error: bool = True) -> dict[str, Path]:
    """Fetch every configured macro series and write a Parquet per series.

    Args:
        skip_on_error: If True, log and skip any failing series rather than
            aborting. Useful when running in a partially networked environment.

    Returns:
        Mapping ``series_name -> parquet_path`` for every successfully saved series.
    """
    saved: dict[str, Path] = {}
    for cfg in _series_configs():
        try:
            df = fetch_series(cfg)
            saved[cfg.name] = save_series(df, cfg.name)
        except Exception as e:  # noqa: BLE001
            if not skip_on_error:
                raise
            logger.error("Failed to fetch %s: %s", cfg.name, e)
    return saved
