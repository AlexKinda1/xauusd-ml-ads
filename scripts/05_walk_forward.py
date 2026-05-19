"""Walk-forward + h=4 experiment — runs both XGBoost (with refit) and
Chronos (zero-shot with sliding context) on the same monthly fold grid.

Designed to run on Colab; the sandbox can run only the XGBoost half.

Pipeline:
  1. Load the aligned dataset (Phase 1 output).
  2. Build features + multi-horizon targets (h=1, h=4, h=24).
  3. Define monthly folds covering the historical test window
     (sept 2023 → end of data).
  4. For each fold:
       - Refit XGBoost on all data BEFORE the fold (minus an embargo of h).
       - Run Chronos zero-shot with the last 256 bars as context.
       - Save fold predictions.
  5. Aggregate predictions, compute metrics + simple long-short Sharpe.

Usage::

    python scripts/05_walk_forward.py --horizon 4 \
        --fold-size 720 --chronos-model amazon/chronos-bolt-base \
        --skip-chronos      # if running in sandbox
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import visualizations as vz
from src.evaluation.metrics_regression import regression_metrics
from src.features import calendar as cal
from src.features import macro as macro_feat
from src.features import sentiment as sent_feat
from src.features import target as target_mod
from src.features import technical as tech
from src.models.xgboost_model import XGBoostRegressor
from src.training.walk_forward import walk_forward_predictions
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PRED_DIR = PROJECT_ROOT / "data/processed/predictions"
FIG_DIR = PROJECT_ROOT / "reports/figures/walkforward"


def build_dataset(horizon: int) -> tuple[pd.DataFrame, list[str], str]:
    """Build the feature dataset with the requested horizon target."""
    aligned = pd.read_parquet(PROJECT_ROOT / "data/processed/dataset_aligned.parquet")
    feature_pieces = [
        aligned[["open", "high", "low", "close", "volume"]],
        tech.build_technical_features(aligned, feature_lag=1),
        macro_feat.build_macro_features(aligned, feature_lag=1),
        sent_feat.build_sentiment_features(aligned, feature_lag=1),
        cal.calendar_features(aligned.index),
        target_mod.build_targets(aligned["close"], horizon=horizon, vol_window=24,
                                 threshold_factor=0.5),
    ]
    df = pd.concat(feature_pieces, axis=1)
    df = df.iloc[max(168, 200):]
    target_col = f"y_reg_h{horizon}"
    feature_cols = [c for c in df.columns
                    if c not in {"open", "high", "low", "close", "volume",
                                 target_col, f"y_clf_h{horizon}", "y_clf_threshold"}]
    return df, feature_cols, target_col


def naive_sharpe(y_pred: np.ndarray, y_true: np.ndarray, h: int) -> dict:
    """Annualised Sharpe of a naïve long-short strategy: long if pred>0, short if pred<0."""
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    pred, true = y_pred[mask], y_true[mask]
    position = np.sign(pred)               # 1, -1, or 0
    pnl = position * true                  # log-return of the position
    if pnl.std() == 0:
        return {"mean": 0.0, "std": 0.0, "sharpe_annual": 0.0,
                "win_rate": 0.0, "n_trades": int((position != 0).sum())}
    # Each prediction covers h hours; per-year = 24*252 / h ≈ trade horizons / yr
    periods_per_year = 24 * 252 / max(h, 1)
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))
    return {
        "mean": float(pnl.mean()), "std": float(pnl.std()),
        "sharpe_annual": sharpe,
        "win_rate": float((pnl > 0).sum() / max((pnl != 0).sum(), 1)),
        "n_trades": int((position != 0).sum()),
    }


def run_xgboost_walkforward(
    df: pd.DataFrame, feature_cols: list[str], target_col: str,
    horizon: int, initial_train_end_idx: int, fold_size: int,
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Walk-forward XGBoost with full refit at each fold."""
    def factory() -> XGBoostRegressor:
        return XGBoostRegressor(
            feature_cols=feature_cols,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )

    def fit_kwargs(train_df: pd.DataFrame, fold_df: pd.DataFrame) -> dict:
        # Use the last 10% of train as internal validation for early stopping.
        split = int(len(train_df) * 0.9)
        val = train_df.iloc[split:]
        return {"X_val": val, "y_val": val[target_col].values}

    preds = walk_forward_predictions(
        df, feature_cols=feature_cols, target_col=target_col,
        model_factory=factory,
        initial_train_end_idx=initial_train_end_idx,
        fold_size=fold_size, embargo=horizon, n_folds=n_folds,
        extra_cols=["close"], fit_kwargs_factory=fit_kwargs,
    )
    return preds


def run_chronos_walkforward(
    df: pd.DataFrame, target_col: str, horizon: int,
    initial_train_end_idx: int, fold_size: int,
    model_id: str = "amazon/chronos-bolt-base",
    context_length: int = 256,
    batch_size: int = 32,
    device: str = "cuda",
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Chronos zero-shot predictions over the same fold grid as XGBoost.

    Note: Chronos is zero-shot — there is no per-fold refit. The 'walk-forward'
    nature is achieved by always using the most-recent ``context_length`` bars
    when predicting each row. This naturally keeps the prediction context in
    the current regime.
    """
    from src.models.chronos_model import ChronosConfig, ChronosRegressor

    cfg = ChronosConfig(
        pretrained=model_id, context_length=context_length,
        horizon=horizon, num_samples=20, batch_size=batch_size, device=device,
    )
    model = ChronosRegressor(cfg=cfg)
    model.fit(df, df[target_col].values)   # loads pipeline only

    chunks = []
    fold_start = initial_train_end_idx
    fold_idx = 0
    while fold_start + fold_size <= len(df):
        if n_folds is not None and fold_idx >= n_folds:
            break
        fold_end = fold_start + fold_size
        fold_df = df.iloc[fold_start:fold_end]
        logger.info("Chronos fold %d | %s -> %s | %d rows",
                    fold_idx, fold_df.index[0], fold_df.index[-1], len(fold_df))
        # Provide enough context: include the (context_length-1) rows before fold_start
        ctx_start = max(0, fold_start - context_length + 1)
        sub = df.iloc[ctx_start:fold_end]
        preds = model.predict(sub)
        # Keep only the predictions corresponding to fold_df rows.
        preds_for_fold = preds[-len(fold_df):]
        chunk = pd.DataFrame(
            {"y_true": fold_df[target_col].values, "y_pred": preds_for_fold,
             "close": fold_df["close"].values, "fold": fold_idx},
            index=fold_df.index,
        )
        chunks.append(chunk)
        fold_start = fold_end
        fold_idx += 1
    return pd.concat(chunks, axis=0)


def make_figures(preds: pd.DataFrame, model_name: str, horizon: int) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / f"{model_name}_h{horizon}"
    y_true = preds["y_true"].values
    y_pred = preds["y_pred"].values
    out = {
        "pred_vs_actual_scatter": vz.predicted_vs_actual_scatter(
            y_true, y_pred, out=base.with_name(f"{base.name}_scatter.png"),
            title=f"{model_name} walk-forward h={horizon} — test predictions"),
        "pred_vs_actual_ts": vz.pred_vs_actual_timeseries(
            y_true, y_pred, preds.index,
            out=base.with_name(f"{base.name}_ts.png"),
            title=f"{model_name} walk-forward h={horizon} — over time",
            downsample=4 if horizon == 4 else 24),
        "returns_scatter": vz.returns_scatter(
            y_true, y_pred, out=base.with_name(f"{base.name}_returns.png"),
            title=f"{model_name} walk-forward h={horizon}"),
        "residuals_histogram": vz.residuals_histogram(
            y_true, y_pred, out=base.with_name(f"{base.name}_residuals_hist.png"),
            title=f"{model_name} walk-forward h={horizon} — residuals"),
        "residuals_over_time": vz.residuals_over_time(
            y_true, y_pred, preds.index,
            out=base.with_name(f"{base.name}_residuals_ts.png"),
            title=f"{model_name} walk-forward h={horizon} — residuals over time",
            downsample=4 if horizon == 4 else 24),
        "monthly_dir_acc": vz.monthly_directional_accuracy(
            y_true, y_pred, preds.index,
            out=base.with_name(f"{base.name}_monthly_diracc.png"),
            title=f"{model_name} walk-forward h={horizon} — monthly directional accuracy"),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--fold-size", type=int, default=720,
                        help="bars per fold (default 720 = ~30 days for h=4 in H1)")
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--initial-train-cutoff", default="2023-09-05",
                        help="ISO date; first fold starts here")
    parser.add_argument("--chronos-model", default="amazon/chronos-bolt-base")
    parser.add_argument("--chronos-context", type=int, default=256)
    parser.add_argument("--chronos-batch", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-chronos", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    args = parser.parse_args()

    set_global_seed(42)
    h = args.horizon
    logger.info("=== Walk-forward experiment | horizon=%d | fold_size=%d ===", h, args.fold_size)

    df, feature_cols, target_col = build_dataset(h)
    logger.info("Dataset shape: %s, features: %d, target: %s",
                df.shape, len(feature_cols), target_col)

    cutoff = pd.Timestamp(args.initial_train_cutoff, tz="UTC")
    initial_idx = df.index.get_indexer([cutoff], method="nearest")[0]
    logger.info("Initial train ends at %s (idx %d)", df.index[initial_idx], initial_idx)

    summary: dict = {"horizon": h, "fold_size": args.fold_size,
                     "n_features": len(feature_cols),
                     "initial_train_cutoff": str(df.index[initial_idx]),
                     "models": {}}

    if not args.skip_xgboost:
        logger.info(">>> XGBoost walk-forward")
        xgb_preds = run_xgboost_walkforward(
            df, feature_cols, target_col, horizon=h,
            initial_train_end_idx=initial_idx, fold_size=args.fold_size,
            n_folds=args.n_folds,
        )
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        xgb_preds.to_parquet(PRED_DIR / f"xgboost_walkforward_h{h}.parquet")
        xgb_metrics = regression_metrics(xgb_preds["y_true"].values, xgb_preds["y_pred"].values)
        xgb_sharpe = naive_sharpe(xgb_preds["y_pred"].values, xgb_preds["y_true"].values, h)
        xgb_figs = make_figures(xgb_preds, "xgboost_wf", h)
        summary["models"]["xgboost_walkforward"] = {
            "n_folds": int(xgb_preds["fold"].nunique()),
            "n_predictions": int(len(xgb_preds)),
            "metrics": xgb_metrics, "sharpe": xgb_sharpe,
            "figures": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in xgb_figs.items()},
        }
        logger.info("XGBoost wf: %s", " | ".join(f"{k}={v:.4f}" for k, v in xgb_metrics.items()))
        logger.info("XGBoost wf Sharpe: %s", xgb_sharpe)

    if not args.skip_chronos:
        logger.info(">>> Chronos walk-forward")
        ch_preds = run_chronos_walkforward(
            df, target_col, horizon=h,
            initial_train_end_idx=initial_idx, fold_size=args.fold_size,
            model_id=args.chronos_model, context_length=args.chronos_context,
            batch_size=args.chronos_batch, device=args.device, n_folds=args.n_folds,
        )
        ch_preds.to_parquet(PRED_DIR / f"chronos_walkforward_h{h}.parquet")
        ch_metrics = regression_metrics(ch_preds["y_true"].values, ch_preds["y_pred"].values)
        ch_sharpe = naive_sharpe(ch_preds["y_pred"].values, ch_preds["y_true"].values, h)
        ch_figs = make_figures(ch_preds, "chronos_wf", h)
        summary["models"]["chronos_walkforward"] = {
            "model_id": args.chronos_model,
            "context_length": args.chronos_context,
            "n_folds": int(ch_preds["fold"].nunique()),
            "n_predictions": int(len(ch_preds)),
            "metrics": ch_metrics, "sharpe": ch_sharpe,
            "figures": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in ch_figs.items()},
        }
        logger.info("Chronos wf: %s", " | ".join(f"{k}={v:.4f}" for k, v in ch_metrics.items()))
        logger.info("Chronos wf Sharpe: %s", ch_sharpe)

    out = PROJECT_ROOT / f"reports/tables/phase5_walkforward_h{h}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Walk-forward summary -> %s", out)


if __name__ == "__main__":
    main()
