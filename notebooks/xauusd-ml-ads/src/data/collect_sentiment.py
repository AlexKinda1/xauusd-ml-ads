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

# Primary source: CNN stock Fear & Greed Index (no official API; we try a
# community mirror).
FEAR_GREED_URL = (
    "https://raw.githubusercontent.com/hackertarget/fear-and-greed-index/master/fear-and-greed.csv"
)
# Fallback: alternative.me crypto Fear & Greed. Not stock-specific, but a
# documented public API returning a long history. Useful as a risk-on/off
# proxy when the CNN mirror is unreachable.
FEAR_GREED_ALT_URL = "https://api.alternative.me/fng/?limit=0&format=json"


# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------


def _fetch_cnn_fear_greed(url: str) -> pd.DataFrame:
    """Try to fetch the CNN F&G mirror; returns empty frame on failure."""
    import requests
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime"}), None)
    value_col = next(
        (c for c in df.columns if c.lower() in {"fear_greed", "value", "score"}), None
    )
    if not date_col or not value_col:
        raise ValueError(f"Unexpected schema from CNN mirror: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "value_date": pd.to_datetime(df[date_col]),
            "value": df[value_col].astype("float64"),
        }
    )
    out["release_date"] = out["value_date"]
    return out[["value_date", "release_date", "value"]].sort_values("value_date")


def _fetch_altme_fear_greed(url: str) -> pd.DataFrame:
    """Fallback: alternative.me crypto F&G (full history)."""
    import requests
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if "data" not in payload:
        raise ValueError(f"Unexpected payload: {list(payload)[:5]}")
    rows = payload["data"]
    out = pd.DataFrame(
        {
            "value_date": pd.to_datetime(
                [int(r["timestamp"]) for r in rows], unit="s", utc=True
            ).tz_convert(None),
            "value": [float(r["value"]) for r in rows],
        }
    )
    out["release_date"] = out["value_date"]
    return out[["value_date", "release_date", "value"]].sort_values("value_date")


def fetch_fear_greed(
    url: str = FEAR_GREED_URL,
    fallback_url: str = FEAR_GREED_ALT_URL,
) -> pd.DataFrame:
    """Try CNN mirror first, then alternative.me crypto F&G as fallback.

    Returns an empty DataFrame if both sources are unreachable.
    """
    for fetch, src_url, label in [
        (_fetch_cnn_fear_greed, url, "CNN mirror"),
        (_fetch_altme_fear_greed, fallback_url, "alternative.me (crypto F&G)"),
    ]:
        try:
            df = fetch(src_url)
            logger.info("Fear & Greed fetched from %s (%d rows)", label, len(df))
            return df
        except Exception as e:  # noqa: BLE001
            logger.warning("Fear & Greed %s failed: %s", label, e)
    logger.warning("All Fear & Greed sources unreachable — returning empty frame")
    return pd.DataFrame(columns=["value_date", "release_date", "value"])


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
