"""Smoke tests for CNN1DRegressor — quick forward/backward and save/load."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.cnn import CNN1DConfig, CNN1DRegressor   # noqa: E402


def test_cnn_fits_one_epoch_and_predicts() -> None:
    rs = np.random.RandomState(0)
    n, lookback, n_feat = 64, 24, 5
    X = rs.normal(0, 1, (n, lookback, n_feat)).astype("float32")
    # Signal: target depends on the mean of the first feature over the window.
    y = X[:, :, 0].mean(axis=1).astype("float32")
    cfg = CNN1DConfig(epochs=2, batch_size=16, learning_rate=5e-3,
                      early_stopping_patience=10, seed=0)
    m = CNN1DRegressor(n_features=n_feat, cfg=cfg).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()
    # After 2 epochs on a small signal, residuals should be finite — the
    # network does not need to outperform anything, just to run end-to-end.


def test_cnn_save_load_roundtrip() -> None:
    rs = np.random.RandomState(0)
    n_feat = 4
    X = rs.normal(0, 1, (32, 16, n_feat)).astype("float32")
    y = X.mean(axis=(1, 2)).astype("float32")
    cfg = CNN1DConfig(epochs=1, batch_size=16, seed=0)
    m = CNN1DRegressor(n_features=n_feat, cfg=cfg).fit(X, y)
    before = m.predict(X)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cnn.pt"
        m.save(p)
        loaded = CNN1DRegressor.load(p)
    after = loaded.predict(X)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
