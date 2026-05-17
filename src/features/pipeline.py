"""End-to-end feature pipeline.

Inputs : aligned dataset from Phase 1 (``data/processed/dataset_aligned.parquet``).
Outputs: feature + target dataset (``data/processed/features_targets.parquet``).

The pipeline is intentionally simple — concat the per-module feature
DataFrames, attach the targets, and drop rows with all-NaN features. Splits
and scaling come in Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.features import calendar, macro, sentiment, target, technical
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_features_and_targets(
    aligned: pd.DataFrame,
    *,
    horizon: int,
    lookback: int,
    feature_lag: int = 1,
    threshold_factor: float = 0.5,
    vol_window: int = 24,
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline on an aligned H1 dataset.

    Args:
        aligned: OHLCV + optional macro/sentiment columns, H1-indexed.
        horizon: Prediction horizon ``h`` in bars.
        lookback: Maximum lookback ``L`` — used only to compute the row
            cut-off below which rows are dropped due to insufficient history.
        feature_lag: How many bars to push features forward (strict anti-leakage).
        threshold_factor: Multiplier for the classification neutral zone.
        vol_window: Window for the rolling-vol estimate that defines the threshold.

    Returns:
        DataFrame indexed on ``aligned.index`` containing:
        - the original OHLCV columns (for diagnostics & ARIMA),
        - ``~30`` technical features,
        - ``~3 × n_macro`` macro features (empty if no macro source aligned),
        - ``~3 × n_sentiment`` sentiment features (empty if none),
        - calendar features,
        - ``y_reg_h<h>``, ``y_clf_h<h>``, and the threshold series.
    """
    tech = technical.build_technical_features(aligned, feature_lag=feature_lag)
    mac = macro.build_macro_features(aligned, feature_lag=feature_lag)
    sent = sentiment.build_sentiment_features(aligned, feature_lag=feature_lag)
    cal = calendar.calendar_features(aligned.index)
    tgt = target.build_targets(
        aligned["close"],
        horizon=horizon,
        vol_window=vol_window,
        threshold_factor=threshold_factor,
    )
    ohlcv_cols = aligned[["open", "high", "low", "close", "volume"]]
    feats = pd.concat([ohlcv_cols, tech, mac, sent, cal, tgt], axis=1)

    # Drop the warm-up rows. ``lookback`` should be >= the longest indicator
    # window (we use 168 bars for log-returns / RV); be defensive and pick max.
    min_required = max(lookback, 200)
    n_before = len(feats)
    feats = feats.iloc[min_required:]
    logger.info(
        "Built feature matrix: %d rows (dropped %d warm-up rows), %d columns",
        len(feats), n_before - len(feats), feats.shape[1],
    )
    return feats


def summarise(feats: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """Quick summary dict suitable for the Phase-2 STOP report."""
    feature_cols = [
        c for c in feats.columns
        if c not in {"open", "high", "low", "close", "volume",
                     f"y_reg_h{horizon}", f"y_clf_h{horizon}", "y_clf_threshold"}
    ]
    nan_rates = feats[feature_cols].isna().mean().sort_values(ascending=False)
    clf_col = f"y_clf_h{horizon}"
    counts = feats[clf_col].value_counts(dropna=False).to_dict()
    return {
        "rows": int(len(feats)),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "top_nan_rates": {k: round(float(v), 4) for k, v in nan_rates.head(8).items()},
        "y_clf_class_balance": {str(k): int(v) for k, v in counts.items()},
        "y_reg_mean_bp": round(float(feats[f"y_reg_h{horizon}"].mean() * 1e4), 4),
        "y_reg_std_bp": round(float(feats[f"y_reg_h{horizon}"].std() * 1e4), 4),
    }


def run(aligned_path: str | Path = "data/processed/dataset_aligned.parquet",
        out_path: str | Path = "data/processed/features_targets.parquet") -> pd.DataFrame:
    """Driver: load aligned data, build features + targets, persist Parquet."""
    cfg = load_training_config()
    task = cfg["task"]

    p_in = PROJECT_ROOT / aligned_path if not Path(aligned_path).is_absolute() else Path(aligned_path)
    p_out = PROJECT_ROOT / out_path if not Path(out_path).is_absolute() else Path(out_path)

    logger.info("Loading aligned dataset from %s", p_in)
    aligned = pd.read_parquet(p_in)

    feats = build_features_and_targets(
        aligned,
        horizon=int(task["horizon"]),
        lookback=int(task["lookback"]),
        feature_lag=1,
        threshold_factor=float(task["neutral_threshold_factor"]),
        vol_window=24,
    )
    p_out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(p_out, engine="pyarrow", compression="snappy")
    logger.info("Saved feature dataset (%d rows, %d cols) -> %s",
                len(feats), feats.shape[1], p_out)
    return feats
