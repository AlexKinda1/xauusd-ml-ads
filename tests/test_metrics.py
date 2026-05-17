"""Tests for the regression and classification metric helpers."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics_classification import classification_metrics, confusion
from src.evaluation.metrics_regression import regression_metrics


def test_regression_metrics_perfect() -> None:
    y = np.array([0.01, -0.02, 0.005, 0.0, -0.01])
    m = regression_metrics(y, y.copy())
    assert m["rmse"] == 0
    assert m["mae"] == 0
    assert m["r2"] == 1.0
    assert m["directional_accuracy"] == 1.0


def test_regression_metrics_handles_nan() -> None:
    y_true = np.array([0.01, np.nan, 0.005, 0.0])
    y_pred = np.array([0.01, 0.02, np.nan, 0.0])
    m = regression_metrics(y_true, y_pred)
    # Only 2 rows usable (indices 0 and 3)
    assert m["rmse"] == 0.0
    assert np.isnan(m["pearson"]) or np.isfinite(m["pearson"])


def test_classification_metrics_perfect() -> None:
    y = np.array([-1, 0, 1, -1, 0, 1])
    m = classification_metrics(y, y.copy())
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["mcc"] == 1.0


def test_confusion_shape() -> None:
    y_true = np.array([-1, 0, 1, -1, 0, 1])
    y_pred = np.array([-1, 0, 0, 0, 0, 1])
    cm = confusion(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)
