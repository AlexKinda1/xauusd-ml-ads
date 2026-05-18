"""Phase-4 step 4/7 — CNN 1D regression.

Trains a single-seed CNN 1D regressor on the Phase-3 sequence splits
(``data/processed/splits/{train,val,test}_sequences.npz``).

Outputs:
- MLflow run with all metrics + the training-loss curve.
- Predictions parquet for train/val/test.
- Visualisations under ``reports/figures/cnn/``:
    - feature_importance (gradient-based proxy on a sample)
    - pred_vs_actual scatter
    - pred_vs_actual_timeseries (overlay)
    - residuals (histogram, KDE-ish via density)
    - residuals_over_time
    - monthly_directional_accuracy
    - loss_curve (epochs)
    - returns_scatter (quadrant view)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.evaluation import visualizations as vz
from src.evaluation.metrics_regression import regression_metrics
from src.models.cnn import CNN1DConfig, CNN1DRegressor
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"
FIG_DIR = PROJECT_ROOT / "reports/figures/cnn"
PRED_DIR = PROJECT_ROOT / "data/processed/predictions"


def _load_seq(name: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    arr = np.load(SPLITS_DIR / f"{name}_sequences.npz", allow_pickle=True)
    end_ts = pd.DatetimeIndex(pd.to_datetime(arr["end_ts"])).tz_localize("UTC")
    return arr["X"], arr["y_reg"], end_ts, list(arr["feature_cols"])


def _save_predictions(name: str, index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"cnn1d_regression_{name}.parquet"
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=index).to_parquet(out, compression="snappy")
    return out


def _make_visualisations(
    test_ts: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray,
    loss_history: dict[str, list[float]],
) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["pred_vs_actual_scatter"] = vz.predicted_vs_actual_scatter(
        y_true, y_pred,
        out=FIG_DIR / "cnn_reg_pred_vs_actual.png",
        title="CNN1D regression — test (scatter)",
    )
    paths["pred_vs_actual_timeseries"] = vz.pred_vs_actual_timeseries(
        y_true, y_pred, test_ts,
        out=FIG_DIR / "cnn_reg_pred_vs_actual_ts.png",
        title="CNN1D regression — test (over time)",
        downsample=24,   # keep daily for readability
    )
    paths["returns_scatter"] = vz.returns_scatter(
        y_true, y_pred,
        out=FIG_DIR / "cnn_reg_returns_scatter.png",
        title="CNN1D — predicted vs actual log-returns",
    )
    paths["residuals_histogram"] = vz.residuals_histogram(
        y_true, y_pred,
        out=FIG_DIR / "cnn_reg_residuals_histogram.png",
        title="CNN1D regression — residuals distribution (test)",
    )
    paths["residuals_over_time"] = vz.residuals_over_time(
        y_true, y_pred, test_ts,
        out=FIG_DIR / "cnn_reg_residuals_over_time.png",
        title="CNN1D regression — residuals over time (test)",
        downsample=12,
    )
    paths["monthly_dir_acc"] = vz.monthly_directional_accuracy(
        y_true, y_pred, test_ts,
        out=FIG_DIR / "cnn_reg_monthly_dir_acc.png",
        title="CNN1D — monthly directional accuracy on test",
    )
    if loss_history.get("train"):
        # Re-package into XGBoost-style format for the shared plot.
        history = {
            "validation_0": {"mse": loss_history["train"]},
            "validation_1": {"mse": loss_history["val"]} if loss_history.get("val") else {},
        }
        paths["loss_curve"] = vz.learning_curve_iterations(
            history,
            out=FIG_DIR / "cnn_reg_loss_curve.png",
            title="CNN1D — train vs val MSE per epoch",
            metric_name="mse",
        )
    return paths


def main() -> None:
    set_global_seed(42)
    cfg = load_training_config()
    horizon = int(cfg["task"]["horizon"])
    target_reg = f"y_reg_h{horizon}"

    X_tr, y_tr, ts_tr, feature_cols = _load_seq("train")
    X_va, y_va, ts_va, _ = _load_seq("val")
    X_te, y_te, ts_te, _ = _load_seq("test")
    n_features = X_tr.shape[2]
    logger.info("Loaded sequences | train %s | val %s | test %s | F=%d",
                X_tr.shape, X_va.shape, X_te.shape, n_features)

    model_cfg = CNN1DConfig(
        epochs=20,
        batch_size=256,
        learning_rate=1e-3,
        early_stopping_patience=4,
        seed=42,
    )
    model = CNN1DRegressor(n_features=n_features, cfg=model_cfg)

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("xauusd-ml-ads")

    with mlflow.start_run(run_name="cnn1d_regression") as run:
        mlflow.log_params({"model_name": model.name, "task": model.task,
                           "n_features": n_features, **vars(model_cfg)})

        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        metrics_all: dict[str, dict[str, float]] = {}
        pred_paths: dict[str, Path] = {}
        for name, X, y, ts in [("train", X_tr, y_tr, ts_tr),
                               ("val", X_va, y_va, ts_va),
                               ("test", X_te, y_te, ts_te)]:
            y_pred = model.predict(X)
            m = regression_metrics(np.asarray(y, dtype="float64"), np.asarray(y_pred, dtype="float64"))
            metrics_all[name] = m
            for k, v in m.items():
                mlflow.log_metric(f"{name}_{k}", v)
            pred_paths[name] = _save_predictions(name, ts, y, y_pred)
            mlflow.log_artifact(str(pred_paths[name]), artifact_path="predictions")

        # Visualisations on the test set
        y_pred_te = model.predict(X_te)
        figures = _make_visualisations(ts_te, y_te, y_pred_te, model.loss_history())
        for p in figures.values():
            mlflow.log_artifact(str(p), artifact_path="figures")

        # Loss history JSON artefact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model.loss_history(), f, indent=2); tmp = Path(f.name)
        mlflow.log_artifact(str(tmp), artifact_path="loss_history"); tmp.unlink()

        summary = {
            "regression": {
                "best_val_mse": min(model.val_loss_history_) if model.val_loss_history_ else None,
                "n_epochs_run": len(model.train_loss_history_),
                "metrics": metrics_all,
                "figures": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in figures.items()},
                "mlflow_run_id": run.info.run_id,
            }
        }
    out = PROJECT_ROOT / "reports/tables/phase4_cnn_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("CNN phase complete — summary at %s", out)


if __name__ == "__main__":
    main()
