"""Generic single-model training + MLflow logging loop.

A thin orchestration layer that:
  1. Fits the model on train (with optional val for early stopping).
  2. Predicts on train, val, test.
  3. Computes metrics for the model's task.
  4. Logs everything to MLflow (params, metrics, predictions, model file).

The function is deliberately model-agnostic — it only relies on the
:class:`~src.models.base.ModelBase` API.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.metrics_classification import classification_metrics
from src.evaluation.metrics_regression import regression_metrics
from src.models.base import ModelBase
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    model_name: str
    task: str
    metrics: dict[str, dict[str, float]]   # split -> metrics dict
    predictions_paths: dict[str, Path]
    model_path: Path
    mlflow_run_id: str


def _save_predictions(
    out_dir: Path,
    model_name: str,
    task: str,
    split: str,
    index: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{model_name}_{task}_{split}.parquet"
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=index)
    df.to_parquet(p, engine="pyarrow", compression="snappy")
    return p


def train_and_evaluate(
    model: ModelBase,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    experiment_name: str = "xauusd-ml-ads",
    extra_params: dict[str, Any] | None = None,
    predictions_dir: str | Path = "data/processed/predictions",
    models_dir: str | Path = "data/processed/models",
) -> TrainingResult:
    """Fit ``model``, predict on every split, log to MLflow.

    The DataFrames must share the same column conventions: features + target
    + (optionally) ``close`` for models that need it (ARIMA / Naïve).
    """
    metric_fn = regression_metrics if model.task == "regression" else classification_metrics

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(experiment_name)

    preds_dir = PROJECT_ROOT / predictions_dir
    models_dir_ = PROJECT_ROOT / models_dir

    with mlflow.start_run(run_name=f"{model.name}_{model.task}") as run:
        mlflow.log_params({
            "model_name": model.name,
            "task": model.task,
            "n_features": len(feature_cols),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            **(extra_params or {}),
            **model.params,
        })

        # ARIMA/Naive need the raw close; we pass the entire DataFrame so they
        # can index whatever they need.
        X_train = train_df
        X_val = val_df
        X_test = test_df
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        logger.info("Fitting %s (%s) ...", model.name, model.task)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        metrics: dict[str, dict[str, float]] = {}
        pred_paths: dict[str, Path] = {}
        for split_name, X, y in [("train", X_train, y_train),
                                 ("val", X_val, y_val),
                                 ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            m = metric_fn(np.asarray(y, dtype="float64"), np.asarray(y_pred, dtype="float64"))
            metrics[split_name] = m
            for k, v in m.items():
                mlflow.log_metric(f"{split_name}_{k}", v)

            pred_paths[split_name] = _save_predictions(
                preds_dir, model.name, model.task, split_name,
                index=X.index, y_true=y, y_pred=y_pred,
            )
            mlflow.log_artifact(str(pred_paths[split_name]), artifact_path="predictions")

        model_path = models_dir_ / f"{model.name}_{model.task}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # Persist metrics summary as JSON.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metrics, f, indent=2, default=float)
            tmp_metrics = Path(f.name)
        mlflow.log_artifact(str(tmp_metrics), artifact_path="metrics")
        tmp_metrics.unlink()

        return TrainingResult(
            model_name=model.name,
            task=model.task,
            metrics=metrics,
            predictions_paths=pred_paths,
            model_path=model_path,
            mlflow_run_id=run.info.run_id,
        )
