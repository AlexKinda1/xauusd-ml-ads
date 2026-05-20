"""Phase-5 comparison figures — the full battery.

Generates as many cross-model comparison figures as the available data
allows, into ``reports/figures/comparison/``.

Two data sources:
- h=24 prediction arrays (naive, AR, XGBoost, RF, CNN, BiGRU) for
  array-level comparisons (residuals, equity curves, rolling accuracy...).
- Summary JSONs for the metric bar charts that also cover the TSFMs and
  the walk-forward h=4 experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402
from scipy import stats as ss      # noqa: E402

from src.evaluation import reports as rep   # noqa: E402
from src.evaluation.metrics_regression import regression_metrics   # noqa: E402
from src.utils.config import PROJECT_ROOT   # noqa: E402
from src.utils.logging import get_logger   # noqa: E402

logger = get_logger(__name__)

PRED_DIR = PROJECT_ROOT / "data/processed/predictions"
OUT = PROJECT_ROOT / "reports/figures/comparison"

H24_MODELS = {
    "Naive Zero": "naive_zero_regression_test.parquet",
    "AR": "ar_regression_test.parquet",
    "XGBoost": "xgboost_regression_test.parquet",
    "Random Forest": "random_forest_regression_test.parquet",
    "CNN1D": "cnn1d_regression_test.parquet",
    "BiGRU": "bigru_regression_test.parquet",
}
PALETTE = {
    "Naive Zero": "#888888", "AR": "#aa7722", "XGBoost": "#1f77b4",
    "Random Forest": "#2ca02c", "CNN1D": "#d62728", "BiGRU": "#9467bd",
}


def _load_h24() -> dict[str, pd.DataFrame]:
    out = {}
    for name, fn in H24_MODELS.items():
        p = PRED_DIR / fn
        if p.exists():
            out[name] = pd.read_parquet(p)
    return out


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return p


def fig_metric_bars(preds: dict[str, pd.DataFrame]) -> list[Path]:
    """Bar charts for RMSE, MAE, R2, directional accuracy, Pearson (h=24)."""
    rows = {}
    for name, df in preds.items():
        rows[name] = regression_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
    M = pd.DataFrame(rows).T
    paths = []
    specs = [
        ("rmse", "RMSE (lower=better)", False),
        ("mae", "MAE (lower=better)", False),
        ("r2", "R² (higher=better)", True),
        ("directional_accuracy", "Directional accuracy", True),
        ("pearson", "Pearson correlation", True),
    ]
    for col, title, higher_better in specs:
        s = M[col].sort_values(ascending=higher_better)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [PALETTE.get(m, "steelblue") for m in s.index]
        ax.barh(s.index, s.values, color=colors)
        if col == "directional_accuracy":
            ax.axvline(0.5, color="r", ls="--", lw=0.8, label="random=0.5"); ax.legend()
        if col in ("r2", "pearson"):
            ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"h=24 static — {title}")
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
        paths.append(_save(fig, f"cmp_h24_{col}.png"))
    return paths


def fig_equity_curves(preds: dict[str, pd.DataFrame]) -> Path:
    """Cumulative log-PnL of a naive long-short strategy per model (h=24)."""
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, df in preds.items():
        d = df.dropna()
        pnl = np.sign(d["y_pred"].to_numpy()) * d["y_true"].to_numpy()
        equity = np.cumsum(pnl)
        ax.plot(d.index, equity, label=name, color=PALETTE.get(name), lw=1.1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Equity curves (cumulative log-PnL, naive long-short, h=24 test)")
    ax.set_ylabel("cumulative log-return"); ax.legend(loc="upper left", fontsize=8)
    return _save(fig, "cmp_h24_equity_curves.png")


def fig_residual_kde(preds: dict[str, pd.DataFrame]) -> Path:
    """Overlaid residual densities (h=24)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.linspace(-0.05, 0.05, 400)
    for name, df in preds.items():
        d = df.dropna()
        resid = (d["y_true"] - d["y_pred"]).to_numpy()
        resid = resid[np.abs(resid) < 0.05]
        if len(resid) > 10:
            kde = ss.gaussian_kde(resid)
            ax.plot(xs, kde(xs), label=name, color=PALETTE.get(name), lw=1.2)
    ax.set_title("Residual densities (h=24 test)"); ax.set_xlabel("y_true - y_pred")
    ax.legend(fontsize=8)
    return _save(fig, "cmp_h24_residual_kde.png")


def fig_pred_vs_actual_grid(preds: dict[str, pd.DataFrame]) -> Path:
    """2x3 grid of predicted-vs-actual hexbin scatter (h=24)."""
    n = len(preds)
    ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.2 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, (name, df) in zip(axes, preds.items()):
        d = df.dropna()
        yt, yp = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
        ax.hexbin(yp, yt, gridsize=40, cmap="viridis", mincnt=1)
        lim = max(np.abs(yt).max(), np.abs(yp).max())
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.7)
        ax.set_title(name, fontsize=10); ax.set_xlabel("pred"); ax.set_ylabel("actual")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Predicted vs actual (h=24 test)", fontsize=13)
    return _save(fig, "cmp_h24_pred_vs_actual_grid.png")


def fig_rolling_diracc(preds: dict[str, pd.DataFrame], window: int = 720) -> Path:
    """Rolling directional accuracy over time, all models (h=24)."""
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, df in preds.items():
        d = df.dropna()
        hit = (np.sign(d["y_true"]) == np.sign(d["y_pred"])).astype(float)
        roll = hit.rolling(window, min_periods=window // 2).mean()
        ax.plot(d.index, roll, label=name, color=PALETTE.get(name), lw=1.0)
    ax.axhline(0.5, color="r", ls="--", lw=0.8)
    ax.set_ylim(0.3, 0.7); ax.set_title(f"Rolling directional accuracy ({window}-bar window, h=24 test)")
    ax.set_ylabel("dir. accuracy"); ax.legend(loc="upper left", fontsize=8)
    return _save(fig, "cmp_h24_rolling_diracc.png")


def fig_abs_error_box(preds: dict[str, pd.DataFrame]) -> Path:
    """Boxplot of absolute errors per model (h=24)."""
    data, labels = [], []
    for name, df in preds.items():
        d = df.dropna()
        data.append(np.abs(d["y_true"] - d["y_pred"]).to_numpy())
        labels.append(name)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("Absolute error distribution (h=24 test)"); ax.set_ylabel("|y_true - y_pred|")
    plt.xticks(rotation=20, ha="right")
    return _save(fig, "cmp_h24_abs_error_box.png")


def fig_walkforward_bars() -> list[Path]:
    """Bar charts from the walk-forward summary (h=4): Sharpe, dir_acc, R2, Pearson."""
    wf = rep._load(PROJECT_ROOT / "reports/tables/phase5_walkforward_h4_summary.json")
    rows = []
    for variant, vdata in wf.get("variants", {}).items():
        for model_name, res in vdata.get("models", {}).items():
            m = res.get("metrics", {}); sh = res.get("sharpe", {})
            rows.append({
                "label": f"{model_name}\n{variant}",
                "sharpe": sh.get("sharpe_annual"), "dir_acc": m.get("directional_accuracy"),
                "r2": m.get("r2"), "pearson": m.get("pearson"),
            })
    if not rows:
        return []
    df = pd.DataFrame(rows)
    paths = []
    for col, title in [("sharpe", "Annualised Sharpe"), ("dir_acc", "Directional accuracy"),
                       ("r2", "R²"), ("pearson", "Pearson")]:
        s = df.sort_values(col)
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["seagreen" if v and v > (0.5 if col == "dir_acc" else 0) else "firebrick"
                  for v in s[col]]
        ax.barh(s["label"], s[col], color=colors)
        if col == "dir_acc":
            ax.axvline(0.5, color="r", ls="--", lw=0.8)
        else:
            ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"Walk-forward h=4 — {title}")
        for i, v in enumerate(s[col]):
            if v is not None:
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        paths.append(_save(fig, f"cmp_wf_{col}.png"))
    return paths


def fig_diracc_vs_sharpe() -> Path | None:
    """Scatter dir_acc vs Sharpe for walk-forward models (h=4)."""
    wf = rep._load(PROJECT_ROOT / "reports/tables/phase5_walkforward_h4_summary.json")
    pts = []
    for variant, vdata in wf.get("variants", {}).items():
        for model_name, res in vdata.get("models", {}).items():
            m = res.get("metrics", {}); sh = res.get("sharpe", {})
            pts.append((m.get("directional_accuracy"), sh.get("sharpe_annual"),
                        f"{model_name}/{variant}"))
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    for da, sr, lbl in pts:
        ax.scatter(da, sr, s=60)
        ax.annotate(lbl, (da, sr), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0.5, color="r", ls="--", lw=0.8)
    ax.set_xlabel("Directional accuracy"); ax.set_ylabel("Annualised Sharpe")
    ax.set_title("Walk-forward h=4 — dir. accuracy vs Sharpe")
    return _save(fig, "cmp_wf_diracc_vs_sharpe.png")


def fig_metric_heatmap() -> Path:
    """Normalised metric heatmap across all models (master table)."""
    df = rep.build_master_table()
    df = df.dropna(subset=["rmse"]).copy()
    df["label"] = df["model"]
    metrics = ["rmse", "mae", "r2", "dir_acc", "pearson"]
    mat = df.set_index("label")[metrics].astype(float)
    # Normalise each column 0..1 (invert rmse/mae so higher=better everywhere)
    norm = mat.copy()
    for c in metrics:
        col = mat[c]
        if c in ("rmse", "mae"):
            col = -col
        norm[c] = (col - col.min()) / (col.max() - col.min() + 1e-12)
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=(8, max(5, 0.45 * len(norm))))
    im = ax.imshow(norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(norm))); ax.set_yticklabels(norm.index, fontsize=8)
    for i in range(len(norm)):
        for j, c in enumerate(metrics):
            ax.text(j, i, f"{mat.iloc[i][c]:.3f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.6, label="normalised (green=better)")
    ax.set_title("All models — normalised metric heatmap")
    return _save(fig, "cmp_metric_heatmap.png")


def main() -> None:
    preds = _load_h24()
    logger.info("Loaded %d h=24 prediction sets: %s", len(preds), list(preds))
    generated: list[Path] = []
    if preds:
        generated += fig_metric_bars(preds)
        generated.append(fig_equity_curves(preds))
        generated.append(fig_residual_kde(preds))
        generated.append(fig_pred_vs_actual_grid(preds))
        generated.append(fig_rolling_diracc(preds))
        generated.append(fig_abs_error_box(preds))
    generated += fig_walkforward_bars()
    s = fig_diracc_vs_sharpe()
    if s:
        generated.append(s)
    generated.append(fig_metric_heatmap())
    logger.info("Generated %d comparison figures in %s", len(generated), OUT)
    for p in generated:
        logger.info("  %s", p.name)


if __name__ == "__main__":
    main()
