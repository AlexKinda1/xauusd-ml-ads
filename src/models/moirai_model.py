"""MOIRAI (Salesforce TSFM) wrapper for log-return regression.

Loaded via the official ``uni2ts`` package. ``uni2ts`` ships hard pins on
pandas / numpy / scipy / torch / fsspec that clash with the default Colab
environment; the dedicated Colab notebook installs ``uni2ts`` first and
forces a runtime restart so the downgraded libraries are picked up cleanly.

Same input/output contract as the Chronos wrapper: accepts a DataFrame
with a ``close`` column, returns the predicted h-bar log-return for every
row.
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
class MoiraiConfig:
    pretrained: str = "Salesforce/moirai-1.0-R-base"
    context_length: int = 512
    horizon: int = 4
    num_samples: int = 20
    batch_size: int = 16
    patch_size: int | str = "auto"
    device: str = "auto"
    seed: int = 42


class MoiraiRegressor(ModelBase):
    """Zero-shot MOIRAI forecaster over univariate close prices.

    For each row at time ``t`` we take the last ``context_length`` closes
    ending at ``t``, ask MOIRAI for an ``h``-step forecast, take the median
    of the predictive samples, and convert back to ``log(P_{t+h} / P_t)``.
    """

    name = "moirai"

    def __init__(self, cfg: MoiraiConfig | None = None, **kwargs: Any) -> None:
        super().__init__(task="regression", **kwargs)
        self.cfg = cfg or MoiraiConfig()
        self.params.update(vars(self.cfg))
        self._predictor = None
        self._device = None

    # ------------------------------------------------------------------
    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError as e:
            raise RuntimeError(
                "uni2ts not installed. Run: pip install 'uni2ts[notebook]' "
                "(see notebooks/07_colab_moirai.ipynb for the runtime-restart "
                "workaround on Colab)."
            ) from e
        import torch

        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        logger.info("Loading MOIRAI %s on %s", self.cfg.pretrained, device)

        module = MoiraiModule.from_pretrained(self.cfg.pretrained)
        model = MoiraiForecast(
            module=module,
            prediction_length=self.cfg.horizon,
            context_length=self.cfg.context_length,
            patch_size=self.cfg.patch_size,
            num_samples=self.cfg.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._model = model.to(device)
        self._torch_module = module.to(device)
        self._predictor = model
        return self._predictor

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "MoiraiRegressor":
        self._ensure_predictor()
        return self

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Per-row 24h log-return forecasts using MOIRAI medians."""
        if not isinstance(X, pd.DataFrame) or "close" not in X.columns:
            raise ValueError("MoiraiRegressor expects a DataFrame with a 'close' column.")
        import torch

        self._ensure_predictor()
        cfg = self.cfg
        close = X["close"].to_numpy(dtype="float64")
        n = len(close)
        preds = np.full(n, np.nan, dtype="float64")

        # Build a [B, L+H, 1] tensor batch where the first L columns are the
        # context and the last H columns are the predict-target (filled with
        # zeros, masked out by MOIRAI's prediction-mask convention).
        device = self._device
        bs = cfg.batch_size
        L = cfg.context_length
        H = cfg.horizon

        for chunk_start in range(0, n, bs):
            chunk_end = min(chunk_start + bs, n)
            B = chunk_end - chunk_start
            # Stack contexts
            ctx = np.zeros((B, L, 1), dtype="float32")
            for i in range(B):
                t = chunk_start + i
                start = max(0, t - L + 1)
                window = close[start:t + 1].astype("float32")
                ctx[i, -len(window):, 0] = window
            past_target = torch.tensor(ctx, device=device)
            past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros((B, L), device=device, dtype=torch.bool)
            with torch.no_grad():
                # MOIRAI MoiraiForecast.forward expects these tensors; the
                # output is [B, num_samples, H, 1]
                output = self._model(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad,
                )
            samples = output.cpu().numpy()
            if samples.ndim == 4:
                samples = samples[:, :, :, 0]
            median_path = np.median(samples, axis=1)         # [B, H]
            p_tph = median_path[:, -1]
            p_t = close[chunk_start:chunk_end]
            preds[chunk_start:chunk_end] = np.log(np.clip(p_tph, 1e-12, None) /
                                                  np.clip(p_t, 1e-12, None))
        return preds

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.with_suffix(".json").open("w") as f:
            json.dump({"cfg": vars(self.cfg)}, f, indent=2)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "MoiraiRegressor":
        p = Path(path).with_suffix(".json")
        with p.open("r") as f:
            state = json.load(f)
        return cls(cfg=MoiraiConfig(**state["cfg"]))
