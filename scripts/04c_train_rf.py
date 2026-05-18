"""Phase-4 step 3/7 — Random Forest regression + classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import visualizations as vz
from src.models.random_forest import (
    RandomForestClassifierModel,
    RandomForestRegressorModel,
)
from src.training.hyperparameter import tune
from src.training.trainer import train_and_evaluate
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"
FIG_DIR = PROJECT_ROOT / "reports/figures/random_forest"


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}_tabular.parquet")


def _suggest_rf(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def _make_visualisations(
    model, task: str, feature_cols: list[str],
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    target_col: str,
) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    y_true = test_df[target_col].values
    y_pred = model.predict(test_df)

    paths["feature_importance"] = vz.feature_importance_bar(
        feature_cols, model.feature_importances_,
        out=FIG_DIR / f"rf_{task}_feature_importance.png",
        title=f"Random Forest {task} — top 20 features",
    )

    if task == "regression":
        paths["pred_vs_actual"] = vz.predicted_vs_actual_scatter(
            y_true, y_pred,
            out=FIG_DIR / "rf_reg_pred_vs_actual.png",
            title="Random Forest regression — test predictions",
        )
        paths["residuals"] = vz.residuals_histogram(
            y_true, y_pred,
            out=FIG_DIR / "rf_reg_residuals.png",
            title="Random Forest regression — residuals on test",
        )
        paths["monthly_dir_acc"] = vz.monthly_directional_accuracy(
            y_true, y_pred, test_df.index,
            out=FIG_DIR / "rf_reg_monthly_dir_acc.png",
            title="Random Forest regression — monthly directional accuracy on test",
        )
        # Learning curve over training-set size (slow — small CV)
        try:
            from sklearn.ensemble import RandomForestRegressor
            small_est = RandomForestRegressor(
                **{**model.model_params, "n_estimators": min(model.model_params["n_estimators"], 100)}
            )
            Xt = train_df[feature_cols].fillna(train_df[feature_cols].median(numeric_only=True))
            paths["learning_curve"] = vz.learning_curve_data_size(
                small_est, Xt.values, train_df[target_col].values,
                out=FIG_DIR / "rf_reg_learning_curve.png",
                title="Random Forest regression — learning curve",
                scoring="neg_root_mean_squared_error",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Learning curve failed: %s", e)
    else:
        proba = model.predict_proba(test_df)
        paths["confusion"] = vz.confusion_heatmap(
            y_true, y_pred,
            out=FIG_DIR / "rf_clf_confusion.png",
            title="Random Forest classification — test",
        )
        paths["prob_by_true_class"] = vz.prob_by_true_class(
            y_true, proba,
            out=FIG_DIR / "rf_clf_prob_by_true_class.png",
            title="Random Forest — predicted P() by true class",
        )
        paths["classification_report"] = vz.classification_report_heatmap(
            y_true, y_pred,
            out=FIG_DIR / "rf_clf_report.png",
            title="Random Forest classification — report (test)",
        )
        paths["roc"] = vz.roc_curves(
            y_true, proba,
            out=FIG_DIR / "rf_clf_roc.png",
            title="Random Forest — ROC (one-vs-rest, test)",
        )
        paths["pr"] = vz.precision_recall_curves(
            y_true, proba,
            out=FIG_DIR / "rf_clf_pr.png",
            title="Random Forest — Precision-Recall (test)",
        )
        try:
            from sklearn.ensemble import RandomForestClassifier
            small_est = RandomForestClassifier(
                **{**model.model_params, "n_estimators": min(model.model_params["n_estimators"], 100)}
            )
            Xt = train_df[feature_cols].fillna(train_df[feature_cols].median(numeric_only=True))
            paths["learning_curve"] = vz.learning_curve_data_size(
                small_est, Xt.values, train_df[target_col].astype(int).values,
                out=FIG_DIR / "rf_clf_learning_curve.png",
                title="Random Forest classification — learning curve",
                scoring="neg_log_loss",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Learning curve failed: %s", e)
    return paths


def _train_one(task: str, target_col: str, feature_cols: list[str],
               train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
               n_trials: int) -> dict:
    if task == "regression":
        def build(**p): return RandomForestRegressorModel(feature_cols=feature_cols, **p)
    else:
        def build(**p): return RandomForestClassifierModel(feature_cols=feature_cols, **p)

    logger.info("[%s] Optuna tuning — %d trials", task, n_trials)
    best_params, study = tune(
        suggest_params=_suggest_rf, build_model=build,
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
    vis = _make_visualisations(model, task, feature_cols, train_df, val_df, test_df, target_col)

    return {
        "best_params": best_params,
        "tuning_best_value": float(study.best_value),
        "metrics": res.metrics,
        "model_path": str(res.model_path.relative_to(PROJECT_ROOT)),
        "mlflow_run_id": res.mlflow_run_id,
        "figures": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in vis.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
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
    out = PROJECT_ROOT / "reports/tables/phase4_rf_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Random Forest phase complete — summary at %s", out)


if __name__ == "__main__":
    main()
