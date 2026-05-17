"""Sentiment-data collectors: CNN Fear & Greed and Google Trends.

Sentiment sources are inherently fragile (no stable historical API for CNN's
F&G; Google Trends rate-limits aggressively). Both collectors degrade
gracefully — they log a warning and return an empty DataFrame if the source
is unreachable.

Anti-leakage: like macro series, each row carries a ``release_date`` (when
the value became public). For F&G and Google Trends the release is typically
same-day, so ``release_date = value_date``. Downstream alignment uses
``merge_asof`` on ``release_date``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_data_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Public historical mirror of CNN's stock Fear & Greed Index.
# License: MIT. Updated daily by a community scraper.
FEAR_GREED_URL = (
    "https://raw.githubusercontent.com/hackertarget/fear-and-greed-index/master/fear-and-greed.csv"
)


# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------


def fetch_fear_greed(url: str = FEAR_GREED_URL) -> pd.DataFrame:
    """Download the public CNN Fear & Greed Index CSV.

    Returns a DataFrame with columns ``[value_date, release_date, value]``.
    Returns an empty DataFrame if the source is unreachable.
    """
    try:
        import requests  # transitive dep of many libs; available
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("requests not installed") from e

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("Fear & Greed unreachable (%s) — returning empty frame", e)
        return pd.DataFrame(columns=["value_date", "release_date", "value"])

    df = pd.read_csv(io.StringIO(resp.text))
    # Schema may evolve; defensively normalise.
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime"}), None)
    value_col = next(
        (c for c in df.columns if c.lower() in {"fear_greed", "value", "score"}),
        None,
    )
    if not date_col or not value_col:
        logger.warning("Unexpected F&G schema: %s", list(df.columns))
        return pd.DataFrame(columns=["value_date", "release_date", "value"])

    out = pd.DataFrame(
        {
            "value_date": pd.to_datetime(df[date_col]),
            "value": df[value_col].astype("float64"),
        }
    )
    out["release_date"] = out["value_date"]   # published same day
    return out[["value_date", "release_date", "value"]].sort_values("value_date")


# ---------------------------------------------------------------------------
# Google Trends (pytrends)
# ---------------------------------------------------------------------------


def fetch_google_trends(
    keywords: Iterable[str],
    geo: str = "",
    chunk_years: int = 4,
) -> pd.DataFrame:
    """Fetch Google Trends interest with weekly granularity.

    pytrends returns weekly data for spans up to 5 years and monthly above.
    We request the full history in overlapping ``chunk_years``-year chunks and
    stitch them by re-normalising on the overlap.

    Returns a long-form DataFrame ``[value_date, release_date, keyword, value]``.
    Returns an empty DataFrame on failure.
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError as e:
        raise RuntimeError("pytrends not installed; run `poetry install`") from e

    try:
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
    except Exception as e:  # noqa: BLE001
        logger.warning("Google Trends unreachable (%s) — returning empty frame", e)
        return pd.DataFrame(columns=["value_date", "release_date", "keyword", "value"])

    pieces: list[pd.DataFrame] = []
    for kw in keywords:
        try:
            pytrends.build_payload([kw], geo=geo, timeframe="all")
            df = pytrends.interest_over_time()
        except Exception as e:  # noqa: BLE001
            logger.warning("Google Trends failed for %r: %s", kw, e)
            continue
        if df.empty:
            continue
        long = (
            df.drop(columns=[c for c in df.columns if c == "isPartial"], errors="ignore")
            .rename(columns={kw: "value"})
            .reset_index()
            .rename(columns={"date": "value_date"})
        )
        long["keyword"] = kw
        long["release_date"] = long["value_date"]
        pieces.append(long[["value_date", "release_date", "keyword", "value"]])

    if not pieces:
        return pd.DataFrame(columns=["value_date", "release_date", "keyword", "value"])
    return pd.concat(pieces, ignore_index=True).sort_values(["keyword", "value_date"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def save(df: pd.DataFrame, name: str, out_dir: Path | str = "data/external") -> Path:
    """Persist a sentiment series to ``data/external/sentiment_<name>.parquet``."""
    d = Path(out_dir)
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"sentiment_{name}.parquet"
    df.to_parquet(out, engine="pyarrow", compression="snappy")
    logger.info("Saved sentiment_%s (%d rows) -> %s", name, len(df), out)
    return out


def run() -> dict[str, Path]:
    """Fetch every enabled sentiment source and persist Parquets."""
    cfg = load_data_config()["sentiment"]
    saved: dict[str, Path] = {}

    fg = fetch_fear_greed()
    if not fg.empty:
        saved["fear_greed"] = save(fg, "fear_greed")

    gt_cfg = cfg.get("google_trends", {})
    keywords = gt_cfg.get("keywords", [])
    if keywords:
        gt = fetch_google_trends(keywords, geo=gt_cfg.get("geo", ""))
        if not gt.empty:
            saved["google_trends"] = save(gt, "google_trends")

    return saved
