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


def classification_report_heatmap(
    y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str
) -> Path:
    """Heatmap of sklearn's per-class precision / recall / F1 / support."""
    from sklearn.metrics import classification_report

    mask = ~(np.isnan(y_true.astype("float64")) | np.isnan(y_pred.astype("float64")))
    rpt = classification_report(
        y_true[mask].astype(int), y_pred[mask].astype(int),
        labels=list(CLASS_LABELS), output_dict=True, zero_division=0,
    )
    rows = [str(c) for c in CLASS_LABELS] + ["macro avg", "weighted avg"]
    cols = ["precision", "recall", "f1-score", "support"]
    data = np.array([[rpt[r][c] for c in cols] for r in rows], dtype="float64")
    # Normalise support so it shares the colour scale; show raw in annot.
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1))
    norm = data.copy()
    norm[:, -1] = norm[:, -1] / max(norm[:, -1].max(), 1.0)
    annot = np.array([[f"{data[i, j]:.0f}" if j == 3 else f"{data[i, j]:.3f}"
                       for j in range(4)] for i in range(len(rows))])
    sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=cols, yticklabels=rows, cbar=False, vmin=0, vmax=1)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def roc_curves(
    y_true: np.ndarray, proba: np.ndarray, out: Path, title: str
) -> Path:
    """One-vs-rest ROC curves for the 3 classes."""
    from sklearn.metrics import roc_auc_score, roc_curve
    mask = ~np.isnan(y_true.astype("float64"))
    yt = y_true[mask].astype(int)
    p = proba[mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["firebrick", "gray", "steelblue"]
    for col, cls in enumerate(CLASS_LABELS):
        y_bin = (yt == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, p[:, col])
        auc = roc_auc_score(y_bin, p[:, col])
        ax.plot(fpr, tpr, color=colors[col], lw=1.5, label=f"class {cls}: AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def precision_recall_curves(
    y_true: np.ndarray, proba: np.ndarray, out: Path, title: str
) -> Path:
    """One-vs-rest Precision-Recall curves for the 3 classes."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    mask = ~np.isnan(y_true.astype("float64"))
    yt = y_true[mask].astype(int)
    p = proba[mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["firebrick", "gray", "steelblue"]
    for col, cls in enumerate(CLASS_LABELS):
        y_bin = (yt == cls).astype(int)
        if y_bin.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin, p[:, col])
        ap = average_precision_score(y_bin, p[:, col])
        ax.plot(rec, prec, color=colors[col], lw=1.5,
                label=f"class {cls}: AP={ap:.3f}")
        ax.axhline(y_bin.mean(), color=colors[col], ls=":", lw=0.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def learning_curve_iterations(
    history: dict[str, dict[str, list[float]]],
    out: Path, title: str, metric_name: str | None = None,
) -> Path:
    """XGBoost-style train vs val metric over boosting rounds.

    ``history`` follows xgb.evals_result_ format::

        {"validation_0": {"rmse": [...]}, "validation_1": {"rmse": [...]}}

    where validation_0 is train and validation_1 is val.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    color = {"validation_0": "steelblue", "validation_1": "firebrick"}
    label = {"validation_0": "train", "validation_1": "val"}
    for eval_name, metrics in history.items():
        if not metrics:
            continue
        m_key = metric_name or next(iter(metrics))
        vals = metrics.get(m_key, [])
        if vals:
            ax.plot(range(1, len(vals) + 1), vals,
                    color=color.get(eval_name, "gray"),
                    label=f"{label.get(eval_name, eval_name)} ({m_key})", lw=1.5)
    ax.set_xlabel("Boosting round"); ax.set_ylabel("Metric (lower = better)")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def learning_curve_data_size(
    estimator, X_train, y_train, *, out: Path, title: str,
    scoring: str = "neg_root_mean_squared_error",
    cv_splits: int = 3, train_sizes=None,
) -> Path:
    """sklearn-style learning curve (train/val score vs training-set size).

    Uses a chronological :class:`sklearn.model_selection.TimeSeriesSplit`
    to remain anti-leakage compliant for our time-series data.
    """
    from sklearn.model_selection import TimeSeriesSplit, learning_curve
    import numpy as _np

    if train_sizes is None:
        train_sizes = _np.linspace(0.2, 1.0, 6)

    cv = TimeSeriesSplit(n_splits=cv_splits)
    sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train, cv=cv, scoring=scoring,
        train_sizes=train_sizes, n_jobs=-1, shuffle=False,
    )
    train_mean, train_std = -train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = -val_scores.mean(axis=1), val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sizes, train_mean, "o-", color="steelblue", label="train")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="steelblue")
    ax.plot(sizes, val_mean, "o-", color="firebrick", label="val (CV)")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="firebrick")
    ax.set_xlabel("Training set size"); ax.set_ylabel(f"{scoring}")
    ax.set_title(title); ax.legend()
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
