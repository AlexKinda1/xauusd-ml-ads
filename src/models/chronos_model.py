"""Chronos (Amazon TSFM) wrapper for log-return regression.

Two modes
  - **zero_shot**: pretrained model forecasts ``h`` future close prices given
    the last ``context_length`` close prices. We aggregate the probabilistic
    samples (median) and convert to a 24h log-return.
  - **fine_tuned**: same forecasting head, but we further train the model on
    our train split using LoRA adapters (memory-friendly).

The implementation accepts the raw close-price series as input rather than
the engineered features, since Chronos is designed for univariate time
series. The aligned dataset already has a clean ``close`` column.

Network access is required to download the pretrained weights — this code
is designed to run on Colab / a local machine with HF Hub access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelBase
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChronosConfig:
    pretrained: str = "amazon/chronos-bolt-small"
    context_length: int = 512        # Chronos handles up to 2048; 512 is a robust choice
    horizon: int = 24
    num_samples: int = 20            # Probabilistic samples; median is taken
    batch_size: int = 32
    device: str = "auto"             # "cpu", "cuda", or "auto"
    # Fine-tuning options (used only by ``fit`` when mode='fine_tuned')
    mode: str = "zero_shot"          # "zero_shot" | "fine_tuned"
    fine_tune_epochs: int = 3
    fine_tune_lr: float = 1e-4
    fine_tune_batch_size: int = 16
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    seed: int = 42


class ChronosRegressor(ModelBase):
    """Chronos zero-shot / fine-tuned regressor on the close-price series.

    Expected input ``X``: a DataFrame with at minimum a ``close`` column and
    a chronological index. ``predict(X)`` returns the predicted 24h log-return
    ``log(P_{t+h} / P_t)`` for every row in ``X``.
    """

    name = "chronos"

    def __init__(self, cfg: ChronosConfig | None = None, **kwargs: Any) -> None:
        super().__init__(task="regression", **kwargs)
        self.cfg = cfg or ChronosConfig()
        self.params.update(vars(self.cfg))
        self._pipeline = None    # type: ignore[var-annotated]
        self._device = None       # set on first call

    # ------------------------------------------------------------------
    def _ensure_pipeline(self):
        """Load the Chronos pipeline lazily (network + GPU only on demand)."""
        if self._pipeline is not None:
            return self._pipeline
        try:
            from chronos import ChronosBoltPipeline   # newer chronos-forecasting >= 1.4
        except ImportError:
            try:
                from chronos import ChronosPipeline as ChronosBoltPipeline  # fallback
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
        logger.info("Loading Chronos %s on %s", self.cfg.pretrained, device)
        self._pipeline = ChronosBoltPipeline.from_pretrained(
            self.cfg.pretrained,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        return self._pipeline

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ChronosRegressor":
        """Fit. Zero-shot: nothing to do. Fine-tuned: LoRA on train log-returns."""
        if self.cfg.mode == "zero_shot":
            logger.info("Chronos zero-shot — no training step.")
            self._ensure_pipeline()
            return self
        if self.cfg.mode == "fine_tuned":
            return self._fit_lora(X_train, y_train, X_val, y_val)
        raise ValueError(f"Unknown Chronos mode: {self.cfg.mode!r}")

    def _fit_lora(self, X_train, y_train, X_val=None, y_val=None) -> "ChronosRegressor":
        """LoRA fine-tuning of the Chronos backbone on log-returns regression."""
        # Implementation note: full fine-tuning of Chronos requires the
        # chronos-forecasting "train" pipeline (chronos.training.train),
        # plus peft / LoRA bindings on the T5 backbone. Wiring those in is
        # left as a follow-up if zero-shot is insufficient — kept here as a
        # documented entrypoint with a clear error so the user knows what
        # to install.
        raise NotImplementedError(
            "LoRA fine-tuning requires chronos-forecasting >= 1.5 and a CUDA "
            "GPU. Use cfg.mode='zero_shot' for now and add fine-tuning as a "
            "follow-up step in a Colab notebook with a GPU runtime."
        )

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Predict the 24h log-return for every row in ``X``.

        For each row at time ``t``, we look back ``context_length`` close
        prices ending at ``t`` and ask Chronos to forecast the next
        ``horizon`` prices. We then take the median of the samples and
        compute ``log(median(P_{t+h}) / P_t)`` as the prediction.
        """
        if not isinstance(X, pd.DataFrame) or "close" not in X.columns:
            raise ValueError("ChronosRegressor expects a DataFrame with a 'close' column.")
        import torch

        pipeline = self._ensure_pipeline()
        cfg = self.cfg
        close = X["close"].to_numpy(dtype="float64")
        n = len(X)
        preds = np.full(n, np.nan, dtype="float64")

        # Build (context, current_price) pairs for each row in X.
        # For row i, context = close[max(0, i-context_length+1) : i+1].
        # The very first rows have a context that is shorter than
        # ``context_length`` — Chronos still accepts that.
        starts = np.maximum(0, np.arange(n) - cfg.context_length + 1)
        ends = np.arange(n) + 1
        contexts = [close[s:e].astype("float32") for s, e in zip(starts, ends)]

        # Chronos pipelines accept a list of 1D tensors and batch internally.
        for chunk_start in range(0, n, cfg.batch_size):
            chunk_end = min(chunk_start + cfg.batch_size, n)
            chunk_contexts = [torch.tensor(c) for c in contexts[chunk_start:chunk_end]]
            # forecast: returns [B, num_samples, prediction_length] for Chronos
            #          or [B, num_quantiles, prediction_length] for Chronos-Bolt
            try:
                fc = pipeline.predict(
                    context=chunk_contexts,
                    prediction_length=cfg.horizon,
                    num_samples=cfg.num_samples,
                )
            except TypeError:
                # Chronos-Bolt ignores num_samples — fall back without it
                fc = pipeline.predict(context=chunk_contexts, prediction_length=cfg.horizon)
            fc_np = fc.numpy() if hasattr(fc, "numpy") else np.asarray(fc)
            # Take median over the sample / quantile axis (axis=1).
            median_path = np.median(fc_np, axis=1)              # [B, horizon]
            # We want the price at t+h: last column.
            p_tph = median_path[:, -1]
            p_t = close[chunk_start:chunk_end]
            preds[chunk_start:chunk_end] = np.log(np.clip(p_tph, 1e-12, None) / p_t)
        return preds

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist configuration only — pretrained weights are re-downloaded."""
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.with_suffix(".json").open("w") as f:
            json.dump({"cfg": vars(self.cfg)}, f, indent=2)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ChronosRegressor":
        p = Path(path).with_suffix(".json")
        with p.open("r") as f:
            state = json.load(f)
        return cls(cfg=ChronosConfig(**state["cfg"]))
