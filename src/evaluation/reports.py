"""Aggregate per-phase summary JSONs into the final comparison artefacts.

Produces a master metrics table across every model and experiment, exported
to Markdown, LaTeX and CSV for direct inclusion in the ADS report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)

TABLES_DIR = PROJECT_ROOT / "reports/tables"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Could not parse %s", path)
        return {}


def collect_h24_static() -> list[dict]:
    """Rows for the static h=24 train/test benchmark (Phase 4)."""
    rows: list[dict] = []

    def add(model: str, family: str, metrics_by_split: dict) -> None:
        test = metrics_by_split.get("test", {})
        rows.append({
            "model": model, "family": family, "horizon": 24, "protocol": "static_train_test",
            "rmse": test.get("rmse"), "mae": test.get("mae"), "r2": test.get("r2"),
            "dir_acc": test.get("directional_accuracy"),
            "pearson": test.get("pearson"), "sharpe": None,
        })

    baseline = _load(TABLES_DIR / "phase4_baseline_summary.json")
    if "naive_zero_regression" in baseline:
        add("Naive Zero", "baseline", baseline["naive_zero_regression"]["metrics"])
    if "ar_regression" in baseline:
        add("ARIMA/AR", "baseline", baseline["ar_regression"]["metrics"])

    for fn, model, family in [
        ("phase4_xgboost_summary.json", "XGBoost", "ML-tree"),
        ("phase4_rf_summary.json", "Random Forest", "ML-tree"),
    ]:
        s = _load(TABLES_DIR / fn)
        if "regression" in s:
            add(model, family, s["regression"]["metrics"])

    for fn, model in [("phase4_cnn_summary.json", "CNN 1D"),
                      ("phase4_bigru_summary.json", "BiGRU")]:
        s = _load(TABLES_DIR / fn)
        if "regression" in s:
            add(model, "DL", s["regression"]["metrics"])

    ch = _load(TABLES_DIR / "phase4_chronos_summary.json")
    if "regression_zero_shot" in ch:
        add("Chronos-Bolt (zero-shot)", "TSFM", ch["regression_zero_shot"]["metrics"])
    fc = _load(TABLES_DIR / "phase4_fincast_summary.json")
    if "regression_zero_shot" in fc:
        add("Chronos-T5-Large (zero-shot)", "TSFM", fc["regression_zero_shot"]["metrics"])

    return rows


def collect_walkforward() -> list[dict]:
    """Rows for the walk-forward h=4 experiment (Phase 5)."""
    rows: list[dict] = []
    wf = _load(TABLES_DIR / "phase5_walkforward_h4_summary.json")
    for variant, vdata in wf.get("variants", {}).items():
        for model_name, res in vdata.get("models", {}).items():
            m = res.get("metrics", {})
            sh = res.get("sharpe", {})
            rows.append({
                "model": f"{model_name} ({variant})", "family": _family_of(model_name),
                "horizon": 4, "protocol": f"walkforward_{variant}",
                "rmse": m.get("rmse"), "mae": m.get("mae"), "r2": m.get("r2"),
                "dir_acc": m.get("directional_accuracy"), "pearson": m.get("pearson"),
                "sharpe": sh.get("sharpe_annual"),
                "win_rate": sh.get("win_rate"), "n_trades": sh.get("n_trades"),
            })
    return rows


def _family_of(model_name: str) -> str:
    if "xgboost" in model_name:
        return "ML-tree"
    if "chronos" in model_name or "moirai" in model_name or "fincast" in model_name:
        return "TSFM"
    return "other"


def build_master_table() -> pd.DataFrame:
    rows = collect_h24_static() + collect_walkforward()
    df = pd.DataFrame(rows)
    cols = ["model", "family", "horizon", "protocol", "rmse", "mae", "r2",
            "dir_acc", "pearson", "sharpe", "win_rate", "n_trades"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def export_table(df: pd.DataFrame, stem: str = "phase5_master_comparison") -> dict[str, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    csv = TABLES_DIR / f"{stem}.csv"
    df.to_csv(csv, index=False)
    paths["csv"] = csv

    md = TABLES_DIR / f"{stem}.md"
    md.write_text(df.to_markdown(index=False, floatfmt=".4f"))
    paths["md"] = md

    tex = TABLES_DIR / f"{stem}.tex"
    tex.write_text(df.to_latex(index=False, float_format="%.4f", na_rep="--"))
    paths["latex"] = tex
    return paths
