"""Regression metrics for the h-step log-return target."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard regression metrics + directional accuracy.

    Directional accuracy is the fraction of predictions whose sign matches
    the realised sign — the most actionable metric for an ML/DL trader.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {k: float("nan") for k in
                ("rmse", "mae", "r2", "directional_accuracy", "pearson", "spearman")}

    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(yt, yp))
    # Mean Absolute Percentage Error — robust definition: |y_true - y_pred| / |y_true|
    # restricted to non-zero true values to avoid blow-ups on log-return = 0.
    nonzero = np.abs(yt) > 1e-12
    mape = float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero]))) if nonzero.any() else float("nan")
    r2 = float(r2_score(yt, yp))
    dir_acc = float(np.mean(np.sign(yt) == np.sign(yp)))
    pearson = float(stats.pearsonr(yt, yp)[0]) if yt.std() > 0 and yp.std() > 0 else float("nan")
    spearman = float(stats.spearmanr(yt, yp)[0]) if yt.std() > 0 and yp.std() > 0 else float("nan")
    # Bias = mean residual; informative for systematic over/under-prediction.
    bias = float(np.mean(yp - yt))
    return {
        "mse": mse, "rmse": rmse, "mae": mae, "mape": mape,
        "r2": r2, "directional_accuracy": dir_acc,
        "pearson": pearson, "spearman": spearman, "bias": bias,
    }
