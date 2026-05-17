"""Plotting helpers used by the per-model training scripts.

Every function takes paths in / paths out, never mutates global matplotlib
state, and uses the non-interactive ``Agg`` backend so that the same code
runs on Colab and on a headless CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402
import seaborn as sns              # noqa: E402

from src.evaluation.metrics_classification import confusion, CLASS_LABELS   # noqa: E402

_PALETTE = {"train": "steelblue", "val": "darkorange", "test": "firebrick"}


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def predicted_vs_actual_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str
) -> Path:
    """Hex-bin (predicted, actual) scatter with a diagonal y=x reference."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(yp, yt, gridsize=50, cmap="viridis", mincnt=1)
    lim = max(abs(yt).max(), abs(yp).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(hb, ax=ax, shrink=0.8, label="count")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def residuals_histogram(
    y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str
) -> Path:
    """Histogram of (y_true - y_pred) with a Normal-fit overlay."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    resid = (y_true - y_pred)[mask]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(resid, bins=150, color="steelblue", alpha=0.7, density=True)
    mu, sd = float(resid.mean()), float(resid.std())
    xs = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            "r-", lw=1, label=f"N(μ={mu:.2e}, σ={sd:.2e})")
    ax.set_title(title); ax.legend(); ax.set_xlabel("Residual y_true - y_pred")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def confusion_heatmap(
    y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str
) -> Path:
    """3x3 confusion-matrix heatmap with raw counts and row-normalised pcts."""
    cm = confusion(np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64"))
    rowsums = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = cm / rowsums * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    axes[0].set_title(f"{title} — counts"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    axes[1].set_title(f"{title} — row %"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def prob_by_true_class(
    y_true: np.ndarray, proba: np.ndarray, out: Path, title: str
) -> Path:
    """Box-plot of predicted probabilities for each true class."""
    mask = ~np.isnan(y_true.astype("float64"))
    yt = y_true[mask].astype(int)
    p = proba[mask]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for col, cls in enumerate(CLASS_LABELS):
        data = [p[yt == c, col] for c in CLASS_LABELS]
        axes[col].boxplot(data, tick_labels=[f"true={c}" for c in CLASS_LABELS], showfliers=False)
        axes[col].set_title(f"P(class={cls})"); axes[col].axhline(1/3, color="r", ls="--", lw=0.5)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


def feature_importance_bar(
    feature_names: list[str], importances: np.ndarray, out: Path, title: str, top_n: int = 20
) -> Path:
    """Horizontal bar of the ``top_n`` most-important features."""
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * top_n)))
    ax.barh(np.array(feature_names)[order][::-1], importances[order][::-1], color="steelblue")
    ax.set_title(title); ax.set_xlabel("Importance (gain)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def monthly_directional_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.DatetimeIndex,
    out: Path, title: str,
) -> Path:
    """Bar chart of directional accuracy bucketed by year-month."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=timestamps)
    df = df.dropna()
    df["hit"] = (np.sign(df["y_true"]) == np.sign(df["y_pred"])).astype(int)
    monthly = df["hit"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index, monthly.values, width=20, color="steelblue", alpha=0.8)
    ax.axhline(0.5, color="r", ls="--", lw=0.7, label="random = 0.5")
    ax.set_ylim(0, 1); ax.set_ylabel("Directional accuracy"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def baseline_overlay_metric(
    metrics_by_model: dict[str, dict[str, float]],
    metric_name: str,
    out: Path,
    title: str,
) -> Path:
    """Bar comparing one metric across models (used to highlight 'beats baseline')."""
    names = list(metrics_by_model.keys())
    values = [metrics_by_model[n].get(metric_name, np.nan) for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["gray" if "naive" in n or "majority" in n or n.startswith("ar") else "steelblue" for n in names]
    ax.bar(names, values, color=colors)
    ax.set_title(f"{title} — {metric_name}"); ax.set_ylabel(metric_name)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out
