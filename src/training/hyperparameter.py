"""Optuna-based hyperparameter tuning helper.

Generic enough to drive any :class:`~src.models.base.ModelBase`: callers
provide a ``suggest_params`` function (mapping an Optuna trial to a kwargs
dict), a ``build_model`` factory, and the data.

The objective always minimises a validation-set metric:
- regression : RMSE
- classification : 1 - F1 macro
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd

from src.evaluation.metrics_classification import classification_metrics
from src.evaluation.metrics_regression import regression_metrics
from src.models.base import ModelBase
from src.utils.logging import get_logger

logger = get_logger(__name__)


def tune(
    *,
    suggest_params: Callable[[optuna.Trial], dict[str, Any]],
    build_model: Callable[..., ModelBase],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    task: str,
    n_trials: int = 30,
    seed: int = 42,
    feature_cols: list[str] | None = None,
) -> tuple[dict[str, Any], optuna.Study]:
    """Run ``n_trials`` and return ``(best_params, study)``."""
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        # ``build_model`` already knows ``feature_cols`` (captured in its closure).
        model = build_model(**params)
        model.fit(train_df, y_train, X_val=val_df, y_val=y_val)
        y_pred = model.predict(val_df)
        if task == "regression":
            m = regression_metrics(np.asarray(y_val, dtype="float64"), np.asarray(y_pred, dtype="float64"))
            return float(m["rmse"])
        m = classification_metrics(np.asarray(y_val, dtype="float64"), np.asarray(y_pred, dtype="float64"))
        return float(1.0 - m["f1_macro"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best %s objective: %.6f (params=%s)", task, study.best_value, study.best_params)
    return study.best_params, study
