"""Phase-4 step 5/7 — BiGRU regression.

Mirror of scripts/04d_train_cnn.py: single-seed regression-only training
on the Phase-3 sequence splits, with the full visualisation panel.
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
from src.models.bigru import BiGRUConfig, BiGRURegressor
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"
FIG_DIR = PROJECT_ROOT / "reports/figures/bigru"
PRED_DIR = PROJECT_ROOT / "data/processed/predictions"


def _load_seq(name: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    arr = np.load(SPLITS_DIR / f"{name}_sequences.npz", allow_pickle=True)
    end_ts = pd.DatetimeIndex(pd.to_datetime(arr["end_ts"])).tz_localize("UTC")
    return arr["X"], arr["y_reg"], end_ts, list(arr["feature_cols"])


def _save_predictions(name: str, index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"bigru_regression_{name}.parquet"
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=index).to_parquet(out, compression="snappy")
    return out


def _make_visualisations(
    test_ts: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray,
    loss_history: dict[str, list[float]],
) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "pred_vs_actual_scatter": vz.predicted_vs_actual_scatter(
            y_true, y_pred,
            out=FIG_DIR / "bigru_reg_pred_vs_actual.png",
            title="BiGRU regression — test (scatter)",
        ),
        "pred_vs_actual_timeseries": vz.pred_vs_actual_timeseries(
            y_true, y_pred, test_ts,
            out=FIG_DIR / "bigru_reg_pred_vs_actual_ts.png",
            title="BiGRU regression — test (over time)",
            downsample=24,
        ),
        "returns_scatter": vz.returns_scatter(
            y_true, y_pred,
            out=FIG_DIR / "bigru_reg_returns_scatter.png",
            title="BiGRU — predicted vs actual log-returns",
        ),
        "residuals_histogram": vz.residuals_histogram(
            y_true, y_pred,
            out=FIG_DIR / "bigru_reg_residuals_histogram.png",
            title="BiGRU regression — residuals distribution (test)",
        ),
        "residuals_over_time": vz.residuals_over_time(
            y_true, y_pred, test_ts,
            out=FIG_DIR / "bigru_reg_residuals_over_time.png",
            title="BiGRU regression — residuals over time (test)",
            downsample=12,
        ),
        "monthly_dir_acc": vz.monthly_directional_accuracy(
            y_true, y_pred, test_ts,
            out=FIG_DIR / "bigru_reg_monthly_dir_acc.png",
            title="BiGRU — monthly directional accuracy on test",
        ),
    }
    if loss_history.get("train"):
        history = {
            "validation_0": {"mse": loss_history["train"]},
            "validation_1": {"mse": loss_history["val"]} if loss_history.get("val") else {},
        }
        paths["loss_curve"] = vz.learning_curve_iterations(
            history,
            out=FIG_DIR / "bigru_reg_loss_curve.png",
            title="BiGRU — train vs val MSE per epoch",
            metric_name="mse",
        )
    return paths


def main() -> None:
    set_global_seed(42)
    cfg = load_training_config()
    horizon = int(cfg["task"]["horizon"])
    target_reg = f"y_reg_h{horizon}"   # noqa: F841 (kept for documentation)

    X_tr, y_tr, ts_tr, feature_cols = _load_seq("train")
    X_va, y_va, ts_va, _ = _load_seq("val")
    X_te, y_te, ts_te, _ = _load_seq("test")
    n_features = X_tr.shape[2]
    logger.info("Loaded sequences | train %s | val %s | test %s | F=%d",
                X_tr.shape, X_va.shape, X_te.shape, n_features)

    model_cfg = BiGRUConfig(
        epochs=10,
        batch_size=128,         # reduced from 256 to fit sandbox memory
        hidden_size=64,         # reduced from 128 for the same reason
        num_layers=1,           # reduced from 2; BiGRU = 2 directions already
        learning_rate=1e-3,
        early_stopping_patience=3,
        seed=42,
    )
    model = BiGRURegressor(n_features=n_features, cfg=model_cfg)

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("xauusd-ml-ads")

    with mlflow.start_run(run_name="bigru_regression") as run:
        mlflow.log_params({"model_name": model.name, "task": model.task,
                           "n_features": n_features, **vars(model_cfg)})

        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        metrics_all: dict[str, dict[str, float]] = {}
        for name, X, y, ts in [("train", X_tr, y_tr, ts_tr),
                               ("val", X_va, y_va, ts_va),
                               ("test", X_te, y_te, ts_te)]:
            y_pred = model.predict(X)
            m = regression_metrics(np.asarray(y, dtype="float64"), np.asarray(y_pred, dtype="float64"))
            metrics_all[name] = m
            for k, v in m.items():
                mlflow.log_metric(f"{name}_{k}", v)
            pred_path = _save_predictions(name, ts, y, y_pred)
            mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        y_pred_te = model.predict(X_te)
        figures = _make_visualisations(ts_te, y_te, y_pred_te, model.loss_history())
        for p in figures.values():
            mlflow.log_artifact(str(p), artifact_path="figures")

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
    out = PROJECT_ROOT / "reports/tables/phase4_bigru_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("BiGRU phase complete — summary at %s", out)


if __name__ == "__main__":
    main()
