"""Phase-4 step 6/7 — Chronos (Amazon TSFM) regression.

Designed to run on Colab / a local machine with HF Hub access. Loads the
tabular splits to recover the raw ``close`` column (Chronos is univariate),
runs zero-shot inference, logs everything to MLflow, and produces the same
visualisation panel as the other regression models.

Usage::

    python scripts/04f_train_chronos.py [--model amazon/chronos-bolt-small]
                                        [--context 512] [--horizon 24]
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
from src.models.chronos_model import ChronosConfig, ChronosRegressor
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"
FIG_DIR = PROJECT_ROOT / "reports/figures/chronos"
PRED_DIR = PROJECT_ROOT / "data/processed/predictions"


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}_tabular.parquet")


def _save_predictions(name: str, index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"chronos_regression_{name}.parquet"
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=index).to_parquet(out, compression="snappy")
    return out


def _make_visualisations(test_df: pd.DataFrame, y_pred: np.ndarray, target_col: str) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_true = test_df[target_col].values
    return {
        "pred_vs_actual_scatter": vz.predicted_vs_actual_scatter(
            y_true, y_pred,
            out=FIG_DIR / "chronos_reg_pred_vs_actual.png",
            title="Chronos regression — test (scatter)",
        ),
        "pred_vs_actual_timeseries": vz.pred_vs_actual_timeseries(
            y_true, y_pred, test_df.index,
            out=FIG_DIR / "chronos_reg_pred_vs_actual_ts.png",
            title="Chronos regression — test (over time)",
            downsample=24,
        ),
        "returns_scatter": vz.returns_scatter(
            y_true, y_pred,
            out=FIG_DIR / "chronos_reg_returns_scatter.png",
            title="Chronos — predicted vs actual log-returns",
        ),
        "residuals_histogram": vz.residuals_histogram(
            y_true, y_pred,
            out=FIG_DIR / "chronos_reg_residuals_histogram.png",
            title="Chronos regression — residuals (test)",
        ),
        "residuals_over_time": vz.residuals_over_time(
            y_true, y_pred, test_df.index,
            out=FIG_DIR / "chronos_reg_residuals_over_time.png",
            title="Chronos regression — residuals over time (test)",
            downsample=12,
        ),
        "monthly_dir_acc": vz.monthly_directional_accuracy(
            y_true, y_pred, test_df.index,
            out=FIG_DIR / "chronos_reg_monthly_dir_acc.png",
            title="Chronos — monthly directional accuracy on test",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
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

    chronos_cfg = ChronosConfig(
        pretrained=args.model,
        context_length=args.context,
        horizon=horizon,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        mode="zero_shot",
    )
    model = ChronosRegressor(cfg=chronos_cfg)

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("xauusd-ml-ads")

    with mlflow.start_run(run_name="chronos_regression_zero_shot") as run:
        mlflow.log_params({"model_name": model.name, "task": model.task, **vars(chronos_cfg)})

        # Zero-shot — no fit time, just predict on each split.
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

        # Visualisations on test
        y_pred_te = model.predict(test_df)
        figures = _make_visualisations(test_df, y_pred_te, target_reg)
        for p in figures.values():
            mlflow.log_artifact(str(p), artifact_path="figures")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({k: str(v) for k, v in vars(chronos_cfg).items()}, f); tmp = Path(f.name)
        mlflow.log_artifact(str(tmp), artifact_path="config"); tmp.unlink()

        summary = {
            "regression_zero_shot": {
                "model_id": args.model,
                "context_length": args.context,
                "metrics": metrics_all,
                "figures": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in figures.items()},
                "mlflow_run_id": run.info.run_id,
            }
        }
    out = PROJECT_ROOT / "reports/tables/phase4_chronos_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Chronos phase complete — summary at %s", out)


if __name__ == "__main__":
    main()
