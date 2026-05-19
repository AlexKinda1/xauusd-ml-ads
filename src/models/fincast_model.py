"""FinCast role — second TSFM data point via the larger Chronos T5 model.

Background
~~~~~~~~~~
We initially targeted MOIRAI (Salesforce, via uni2ts) for the
"finance-aware TSFM" slot. In practice, uni2ts hard-pins old versions
of pandas / numpy / scipy / fsspec / torch that conflict with the rest
of the Colab stack and break several downstream packages. We could not
get a stable install on Colab without manual environment surgery.

The original FinCast paper (Liu et al, 2025) does not yet expose a
public HuggingFace checkpoint either.

To still ship a meaningful second TSFM benchmark, we evaluate
``amazon/chronos-t5-large`` here:

  - Same install path as Chronos-Bolt (already proven), no extra conflict.
  - Genuinely different architecture from Chronos-Bolt evaluated earlier:
    T5 encoder-decoder + scalar tokens vs the distilled direct-multi-step
    Bolt model. Chronos-T5-Large is ~710M params (3.5x larger than
    Bolt-base ~200M).
  - Provides a second, larger TSFM data point for the Phase-5 comparison.

This is documented honestly in the run summary so the ADS report can
mention the pivot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelBase
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FinCastConfig:
    # Default points at a *different* Chronos variant than the one used for
    # the Chronos slot (which was amazon/chronos-bolt-base / -small). The
    # T5-large checkpoint is a different architecture and 3.5x larger.
    pretrained: str = "amazon/chronos-t5-large"
    context_length: int = 512
    horizon: int = 24
    num_samples: int = 20
    batch_size: int = 8        # T5-large needs a smaller batch than Bolt
    device: str = "auto"
    seed: int = 42


class FinCastRegressor(ModelBase):
    """Wraps a Chronos-T5 (or any chronos-forecasting compatible) model.

    Same input/output contract as :class:`src.models.chronos_model.ChronosRegressor`:
    accepts a DataFrame with a ``close`` column, returns the predicted
    24h log-return for every row.
    """

    name = "fincast"

    def __init__(self, cfg: FinCastConfig | None = None, **kwargs: Any) -> None:
        super().__init__(task="regression", **kwargs)
        self.cfg = cfg or FinCastConfig()
        self.params.update(vars(self.cfg))
        self._pipeline = None    # type: ignore[var-annotated]
        self._device = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from chronos import ChronosPipeline
        except ImportError as e:
            raise RuntimeError(
                "chronos-forecasting not installed. "
                "Run: pip install chronos-forecasting"
            ) from e
        import torch

        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        logger.info("Loading Chronos-T5 %s on %s", self.cfg.pretrained, device)
        self._pipeline = ChronosPipeline.from_pretrained(
            self.cfg.pretrained,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        return self._pipeline

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "FinCastRegressor":
        self._ensure_pipeline()
        return self

    def predict(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame) or "close" not in X.columns:
            raise ValueError("FinCastRegressor expects a DataFrame with a 'close' column.")
        import torch

        pipeline = self._ensure_pipeline()
        cfg = self.cfg
        close = X["close"].to_numpy(dtype="float64")
        n = len(X)
        preds = np.full(n, np.nan, dtype="float64")

        starts = np.maximum(0, np.arange(n) - cfg.context_length + 1)
        ends = np.arange(n) + 1
        contexts = [close[s:e].astype("float32") for s, e in zip(starts, ends)]

        for chunk_start in range(0, n, cfg.batch_size):
            chunk_end = min(chunk_start + cfg.batch_size, n)
            chunk_contexts = [torch.tensor(c) for c in contexts[chunk_start:chunk_end]]
            fc = pipeline.predict(
                context=chunk_contexts,
                prediction_length=cfg.horizon,
                num_samples=cfg.num_samples,
            )
            fc_np = fc.numpy() if hasattr(fc, "numpy") else np.asarray(fc)
            median_path = np.median(fc_np, axis=1)
            p_tph = median_path[:, -1]
            p_t = close[chunk_start:chunk_end]
            preds[chunk_start:chunk_end] = np.log(np.clip(p_tph, 1e-12, None) / p_t)
        return preds

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.with_suffix(".json").open("w") as f:
            json.dump({"cfg": vars(self.cfg)}, f, indent=2)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "FinCastRegressor":
        p = Path(path).with_suffix(".json")
        with p.open("r") as f:
            state = json.load(f)
        return cls(cfg=FinCastConfig(**state["cfg"]))
