"""Classification metrics for the ternary direction target."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

CLASS_LABELS = (-1, 0, 1)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, F1 macro, MCC, per-class precision/recall."""
    mask = ~(np.isnan(y_true.astype("float64")) | np.isnan(y_pred.astype("float64")))
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    if len(yt) == 0:
        return {k: float("nan") for k in
                ("accuracy", "f1_macro", "mcc", "precision_macro", "recall_macro")}

    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "precision_macro": float(precision_score(yt, yp, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(yt, yp, labels=CLASS_LABELS, average="macro", zero_division=0)),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """3x3 confusion matrix with rows = true, cols = pred, labels (-1, 0, 1)."""
    mask = ~(np.isnan(y_true.astype("float64")) | np.isnan(y_pred.astype("float64")))
    return confusion_matrix(
        y_true[mask].astype(int), y_pred[mask].astype(int), labels=list(CLASS_LABELS)
    )
