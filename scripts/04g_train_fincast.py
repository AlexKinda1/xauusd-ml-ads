"""Phase-4 step 7/7 — FinCast role via Salesforce MOIRAI.

Same orchestration as 04f_train_chronos.py but for MOIRAI. Runs on a
Colab GPU because the sandbox cannot reach HuggingFace Hub.

Usage::

    python scripts/04g_train_fincast.py \
        --model Salesforce/moirai-1.0-R-base \
        --context 512 --batch-size 16 --device cuda
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.evaluation import visualizations as vz
from src.evaluation.metrics_regression import regression_metrics
from src.models.fincast_model import FinCastConfig, FinCastRegressor
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"
FIG_DIR = PROJECT_ROOT / "reports/figures/fincast"
PRED_DIR = PROJECT_ROOT / "data/processed/predictions"


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}_tabular.parquet")


def _save_predictions(name: str, index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"fincast_regression_{name}.parquet"
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=index).to_parquet(out, compression="snappy")
    return out


def _make_visualisations(test_df: pd.DataFrame, y_pred: np.ndarray, target_col: str) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_true = test_df[target_col].values
    return {
        "pred_vs_actual_scatter": vz.predicted_vs_actual_scatter(
            y_true, y_pred,
            out=FIG_DIR / "fincast_reg_pred_vs_actual.png",
            title="FinCast (MOIRAI) — test (scatter)",
        ),
        "pred_vs_actual_timeseries": vz.pred_vs_actual_timeseries(
            y_true, y_pred, test_df.index,
            out=FIG_DIR / "fincast_reg_pred_vs_actual_ts.png",
            title="FinCast (MOIRAI) — test (over time)",
            downsample=24,
        ),
        "returns_scatter": vz.returns_scatter(
            y_true, y_pred,
            out=FIG_DIR / "fincast_reg_returns_scatter.png",
            title="FinCast — predicted vs actual log-returns",
        ),
        "residuals_histogram": vz.residuals_histogram(
            y_true, y_pred,
            out=FIG_DIR / "fincast_reg_residuals_histogram.png",
            title="FinCast — residuals (test)",
        ),
        "residuals_over_time": vz.residuals_over_time(
            y_true, y_pred, test_df.index,
            out=FIG_DIR / "fincast_reg_residuals_over_time.png",
            title="FinCast — residuals over time (test)",
            downsample=12,
        ),
        "monthly_dir_acc": vz.monthly_directional_accuracy(
            y_true, y_pred, test_df.index,
            out=FIG_DIR / "fincast_reg_monthly_dir_acc.png",
            title="FinCast — monthly directional accuracy on test",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Salesforce/moirai-1.0-R-base")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_global_seed(42)
    cfg = load_training_config()
    horizon = int(cfg["task"]["horizon"])
    target_reg = f"y_reg_h{horizon}"

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")
    logger.info("Loaded splits — train=%d, val=%d, test=%d",
                len(train_df), len(val_df), len(test_df))

    fincast_cfg = FinCastConfig(
        pretrained=args.model,
        context_length=args.context,
        horizon=horizon,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
    model = FinCastRegressor(cfg=fincast_cfg)

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("xauusd-ml-ads")

    with mlflow.start_run(run_name="fincast_regression_zero_shot") as run:
        mlflow.log_params({"model_name": model.name, "task": model.task, **vars(fincast_cfg)})

        model.fit(train_df, train_df[target_reg].values)

        metrics_all: dict[str, dict[str, float]] = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            logger.info("Predicting on %s (%d rows)...", name, len(df))
            y_pred = model.predict(df)
            y_true = df[target_reg].values
            m = regression_metrics(np.asarray(y_true, dtype="float64"),
                                   np.asarray(y_pred, dtype="float64"))
            metrics_all[name] = m
            for k, v in m.items():
                mlflow.log_metric(f"{name}_{k}", v)
            pred_path = _save_predictions(name, df.index, y_true, y_pred)
            mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        y_pred_te = model.predict(test_df)
        figures = _make_visualisations(test_df, y_pred_te, target_reg)
        for p in figures.values():
            mlflow.log_artifact(str(p), artifact_path="figures")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({k: str(v) for k, v in vars(fincast_cfg).items()}, f); tmp = Path(f.name)
        mlflow.log_artifact(str(tmp), artifact_path="config"); tmp.unlink()

        summary = {
            "regression_zero_shot": {
                "model_id": args.model,
                "model_family": "MOIRAI",
                "note": "MOIRAI from Salesforce stands in for the 'FinCast' role: no public FinCast checkpoint at our cutoff. MOIRAI was pretrained on LOTSA including a substantial share of financial time series.",
                "context_length": args.context,
                "metrics": metrics_all,
                "figures": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in figures.items()},
                "mlflow_run_id": run.info.run_id,
            }
        }
    out = PROJECT_ROOT / "reports/tables/phase4_fincast_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("FinCast phase complete — summary at %s", out)


if __name__ == "__main__":
    main()
