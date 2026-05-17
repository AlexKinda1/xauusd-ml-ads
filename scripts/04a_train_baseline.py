"""Phase-4 entry point — Baseline models (Naïve + AR).

Trains 4 baselines:
  - NaiveZeroRegressor  (regression)
  - ARRegressor          (regression)
  - MajorityClassifier   (classification)
  - ARClassifier         (classification)

Each is logged to MLflow with its metrics on train/val/test plus the
predictions parquet artefacts.

Usage::

    poetry run python scripts/04a_train_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.baseline import (
    ARClassifier,
    ARRegressor,
    MajorityClassifier,
    NaiveZeroRegressor,
)
from src.training.trainer import train_and_evaluate
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data/processed/splits"


def _load_split(name: str) -> pd.DataFrame:
    """Load a tabular split parquet and re-attach the raw close from features_targets.

    The tabular parquets only contain scaled features + targets; ARIMA-family
    baselines need the unscaled close. We merge it back from the raw feature
    dataset on the index.
    """
    df = pd.read_parquet(SPLITS_DIR / f"{name}_tabular.parquet")
    # `close` is already included in the tabular parquet (we kept it in Phase 3).
    return df


def main() -> None:
    set_global_seed(42)
    cfg = load_training_config()
    horizon = int(cfg["task"]["horizon"])
    target_reg = f"y_reg_h{horizon}"
    target_clf = f"y_clf_h{horizon}"

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")
    feature_cols = [c for c in train_df.columns
                    if c not in {target_reg, target_clf, "close"}]
    logger.info("Loaded splits — train=%d, val=%d, test=%d, features=%d",
                len(train_df), len(val_df), len(test_df), len(feature_cols))

    results = {}
    for model, target in [
        (NaiveZeroRegressor(), target_reg),
        (ARRegressor(horizon=horizon), target_reg),
        (MajorityClassifier(), target_clf),
        (ARClassifier(horizon=horizon), target_clf),
    ]:
        logger.info("=== %s (%s) ===", model.name, model.task)
        res = train_and_evaluate(
            model,
            train_df=train_df, val_df=val_df, test_df=test_df,
            feature_cols=feature_cols, target_col=target,
            experiment_name="xauusd-ml-ads",
        )
        results[f"{model.name}_{model.task}"] = {
            "metrics": res.metrics,
            "model_path": str(res.model_path.relative_to(PROJECT_ROOT)),
            "mlflow_run_id": res.mlflow_run_id,
        }

    summary_path = PROJECT_ROOT / "reports/tables/phase4_baseline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Baseline phase complete — summary at %s", summary_path)


if __name__ == "__main__":
    main()
