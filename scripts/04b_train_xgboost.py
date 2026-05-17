"""Phase-4 step 2/7 — XGBoost regression + classification.

Pipeline:
  1. Load the Phase-3 tabular splits.
  2. Optuna tuning (default 30 trials) on val for each task.
  3. Refit best model on train, evaluate on val + test, log to MLflow.
  4. Generate the standard model-evaluation visualisations into
     ``reports/figures/xgboost/`` and save predictions to parquet.

Usage::

    poetry run python scripts/04b_train_xgboost.py [--n-trials 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import visualizations as vz
from src.models.xgboost_model import XGBoostClassifier, XGBoostRegressor
from src.training.hyperparameter import tune
from src.training.trainer import train_and_evaluate
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"
FIG_DIR = PROJECT_ROOT / "reports/figures/xgboost"


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}_tabular.parquet")


def _suggest_xgb(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
    }


def _make_visualisations(
    task: str,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    feature_names: list[str],
    importances: np.ndarray,
    target_col: str,
) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    y_true = test_df[target_col].values

    paths["feature_importance"] = vz.feature_importance_bar(
        feature_names, importances,
        out=FIG_DIR / f"xgb_{task}_feature_importance.png",
        title=f"XGBoost {task} — top 20 features (gain)",
    )

    if task == "regression":
        paths["pred_vs_actual"] = vz.predicted_vs_actual_scatter(
            y_true, y_pred,
            out=FIG_DIR / "xgb_reg_pred_vs_actual.png",
            title="XGBoost regression — test predictions",
        )
        paths["residuals"] = vz.residuals_histogram(
            y_true, y_pred,
            out=FIG_DIR / "xgb_reg_residuals.png",
            title="XGBoost regression — residuals on test",
        )
        paths["monthly_dir_acc"] = vz.monthly_directional_accuracy(
            y_true, y_pred, test_df.index,
            out=FIG_DIR / "xgb_reg_monthly_dir_acc.png",
            title="XGBoost regression — monthly directional accuracy on test",
        )
    else:
        paths["confusion"] = vz.confusion_heatmap(
            y_true, y_pred,
            out=FIG_DIR / "xgb_clf_confusion.png",
            title="XGBoost classification — test",
        )
        if proba is not None:
            paths["prob_by_true_class"] = vz.prob_by_true_class(
                y_true, proba,
                out=FIG_DIR / "xgb_clf_prob_by_true_class.png",
                title="XGBoost classification — predicted P() by true class",
            )

    return paths


def _train_one(task: str, target_col: str, feature_cols: list[str],
               train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
               n_trials: int) -> dict:
    if task == "regression":
        def build(**p): return XGBoostRegressor(feature_cols=feature_cols, **p)
    else:
        def build(**p): return XGBoostClassifier(feature_cols=feature_cols, **p)

    logger.info("[%s] Optuna tuning — %d trials", task, n_trials)
    best_params, study = tune(
        suggest_params=_suggest_xgb, build_model=build,
        train_df=train_df, val_df=val_df, target_col=target_col,
        task=task, n_trials=n_trials, feature_cols=feature_cols,
    )

    model = build(**best_params)
    res = train_and_evaluate(
        model,
        train_df=train_df, val_df=val_df, test_df=test_df,
        feature_cols=feature_cols, target_col=target_col,
        experiment_name="xauusd-ml-ads",
        extra_params={"tuning_n_trials": n_trials, "tuning_best_value": study.best_value},
    )

    # Visualisations
    proba = None
    if task == "classification":
        proba = model.predict_proba(test_df)
    y_pred_test = model.predict(test_df)
    vis_paths = _make_visualisations(
        task, test_df, y_pred_test, proba,
        feature_names=feature_cols,
        importances=model.feature_importances_,
        target_col=target_col,
    )

    return {
        "best_params": best_params,
        "tuning_best_value": float(study.best_value),
        "metrics": res.metrics,
        "model_path": str(res.model_path.relative_to(PROJECT_ROOT)),
        "mlflow_run_id": res.mlflow_run_id,
        "figures": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in vis_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    set_global_seed(42)
    cfg = load_training_config()
    horizon = int(cfg["task"]["horizon"])
    target_reg = f"y_reg_h{horizon}"
    target_clf = f"y_clf_h{horizon}"

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")
    feature_cols = [c for c in train_df.columns if c not in {target_reg, target_clf, "close"}]
    logger.info("Loaded — train=%d, val=%d, test=%d, features=%d",
                len(train_df), len(val_df), len(test_df), len(feature_cols))

    summary = {
        "regression": _train_one("regression", target_reg, feature_cols, train_df, val_df, test_df, args.n_trials),
        "classification": _train_one("classification", target_clf, feature_cols, train_df, val_df, test_df, args.n_trials),
    }
    out = PROJECT_ROOT / "reports/tables/phase4_xgboost_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("XGBoost phase complete — summary at %s", out)


if __name__ == "__main__":
    main()
