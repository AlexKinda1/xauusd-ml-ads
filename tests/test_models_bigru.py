"""Smoke tests for BiGRURegressor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.bigru import BiGRUConfig, BiGRURegressor  # noqa: E402


def test_bigru_fits_and_predicts() -> None:
    rs = np.random.RandomState(0)
    n, L, F = 32, 16, 5
    X = rs.normal(0, 1, (n, L, F)).astype("float32")
    y = X[:, :, 0].mean(axis=1).astype("float32")
    cfg = BiGRUConfig(epochs=2, batch_size=16, learning_rate=5e-3,
                      early_stopping_patience=10, hidden_size=16, num_layers=1, seed=0)
    m = BiGRURegressor(n_features=F, cfg=cfg).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()


def test_bigru_save_load_roundtrip() -> None:
    rs = np.random.RandomState(0)
    F = 4
    X = rs.normal(0, 1, (16, 12, F)).astype("float32")
    y = X.mean(axis=(1, 2)).astype("float32")
    cfg = BiGRUConfig(epochs=1, batch_size=8, hidden_size=8, num_layers=1, seed=0)
    m = BiGRURegressor(n_features=F, cfg=cfg).fit(X, y)
    before = m.predict(X)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bigru.pt"
        m.save(p)
        loaded = BiGRURegressor.load(p)
    after = loaded.predict(X)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
