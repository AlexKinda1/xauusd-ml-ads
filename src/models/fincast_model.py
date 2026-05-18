"""FinCast role — TSFM specialised in financial / mixed-domain time series.

The original FinCast paper (Liu et al, 2025) does not currently expose a
public HuggingFace checkpoint that we can rely on. To honour the project's
"finance-aware foundation model" slot, we substitute **MOIRAI** from
Salesforce (Liu et al, 2024, ICML) which is pretrained on the LOTSA
corpus including a substantial share of financial time series. MOIRAI's
architecture is genuinely different from Chronos:

  - Chronos: T5 / Chronos-Bolt with scalar quantisation tokens
  - MOIRAI : encoder-only Transformer with multi-patch tokenisation and a
             mixed-distribution probabilistic head

This means we are comparing two distinct TSFM families on XAU/USD, not
two variants of the same one.

If a public FinCast checkpoint becomes available later, swapping is a
one-line change via :attr:`FinCastConfig.pretrained`.
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
    pretrained: str = "Salesforce/moirai-1.0-R-base"
    context_length: int = 512
    horizon: int = 24
    num_samples: int = 20
    batch_size: int = 32
    device: str = "auto"
    patch_size: int | str = "auto"   # MOIRAI auto-selects unless overridden
    seed: int = 42


class FinCastRegressor(ModelBase):
    """MOIRAI-based forecaster, wrapped behind the same ModelBase as Chronos.

    The wrapper accepts a DataFrame with a ``close`` column (univariate
    forecast), runs MOIRAI in zero-shot mode, and returns the predicted
    24h log-return for every row in the input.
    """

    name = "fincast"

    def __init__(self, cfg: FinCastConfig | None = None, **kwargs: Any) -> None:
        super().__init__(task="regression", **kwargs)
        self.cfg = cfg or FinCastConfig()
        self.params.update(vars(self.cfg))
        self._predictor = None    # type: ignore[var-annotated]
        self._device = None

    # ------------------------------------------------------------------
    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError as e:
            raise RuntimeError(
                "uni2ts not installed. Run: pip install 'uni2ts[notebook]'"
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
        self._predictor = model.create_predictor(batch_size=self.cfg.batch_size).to(device)
        return self._predictor

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "FinCastRegressor":
        """Zero-shot — load the predictor lazily and return."""
        self._ensure_predictor()
        return self

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Per-row 24h log-return forecasts using MOIRAI medians."""
        if not isinstance(X, pd.DataFrame) or "close" not in X.columns:
            raise ValueError("FinCastRegressor expects a DataFrame with a 'close' column.")
        import torch
        from gluonts.dataset.pandas import PandasDataset

        predictor = self._ensure_predictor()
        cfg = self.cfg
        close = X["close"].astype("float64")
        n = len(close)
        preds = np.full(n, np.nan, dtype="float64")

        # For efficiency, batch the rows: build a list of mini-series, each
        # containing ``context_length`` closes ending at row i.
        rows = []
        for i in range(n):
            start = max(0, i - cfg.context_length + 1)
            end = i + 1
            sub = close.iloc[start:end].copy()
            sub.index = pd.date_range(
                end=close.index[i] if hasattr(close.index, "to_pydatetime") else f"2020-01-01 {i:04d}:00",
                periods=len(sub), freq="h",
            )
            rows.append((str(i), sub))

        # GluonTS PandasDataset expects a dict-of-Series
        dataset_dict = {item_id: s.rename("target").to_frame() for item_id, s in rows}
        ds = PandasDataset(dataset_dict, target="target", freq="h")
        forecasts = list(predictor.predict(ds))

        # Each forecast has shape [num_samples, prediction_length].
        for fc in forecasts:
            i = int(fc.item_id)
            samples = fc.samples
            median_path = np.median(samples, axis=0)        # [horizon]
            p_tph = median_path[-1]
            p_t = close.iloc[i]
            preds[i] = float(np.log(max(p_tph, 1e-12) / max(p_t, 1e-12)))
        return preds

    # ------------------------------------------------------------------
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
