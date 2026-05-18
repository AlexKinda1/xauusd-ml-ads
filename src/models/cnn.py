"""CNN 1D for log-return regression on sequence windows.

Architecture
  Input  : ``[batch, L=168, F=62]``  (transposed to [B, F, L] for Conv1d)
  Conv1d : channels [64 -> 128 -> 128], kernel_size 3, padding="same"
  BN + ReLU + Dropout after each conv
  Adaptive average pool over the time dimension
  Fully-connected head: 128 -> 64 -> 1  (regression output)

Single seed (per project decision). Early stopping on val RMSE.

Torch is imported lazily so the module can be parsed in environments where
PyTorch is not yet installed.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import ModelBase
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CNN1DConfig:
    conv_channels: list[int] = field(default_factory=lambda: [64, 128, 128])
    kernel_size: int = 3
    dropout: float = 0.3
    fc_hidden: int = 64
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    gradient_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0


def _build_torch_model(n_features: int, cfg: CNN1DConfig):
    """Construct the PyTorch nn.Module. Imported lazily."""
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    in_c = n_features
    for out_c in cfg.conv_channels:
        layers += [
            nn.Conv1d(in_c, out_c, kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        ]
        in_c = out_c
    layers.append(nn.AdaptiveAvgPool1d(1))   # [B, C, 1]
    layers.append(nn.Flatten())              # [B, C]
    layers += [
        nn.Linear(in_c, cfg.fc_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.fc_hidden, 1),
    ]
    return nn.Sequential(*layers)


class CNN1DRegressor(ModelBase):
    """1D CNN regressor over ``[N, L, F]`` sequence windows."""

    name = "cnn1d"

    def __init__(self, n_features: int, cfg: CNN1DConfig | None = None, **kwargs: Any) -> None:
        super().__init__(task="regression", **kwargs)
        self.n_features = n_features
        self.cfg = cfg or CNN1DConfig()
        self.params.update({
            "n_features": n_features,
            **{k: getattr(self.cfg, k) for k in vars(self.cfg)},
        })
        self.model_ = None
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "CNN1DRegressor":  # type: ignore[override]
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.cfg
        rng = torch.Generator(); rng.manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("CNN1D fitting on device=%s | X=%s, y=%s", device, X_train.shape, y_train.shape)

        # X: [N, L, F] -> Conv1d expects [N, F, L], we transpose inside the loop.
        Xt = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)
        train_ds = TensorDataset(Xt, yt)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, generator=rng)

        val_dl = None
        if X_val is not None and y_val is not None:
            Xv = torch.tensor(np.asarray(X_val), dtype=torch.float32)
            yv = torch.tensor(np.asarray(y_val), dtype=torch.float32).view(-1, 1)
            val_dl = DataLoader(TensorDataset(Xv, yv), batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers)

        net = _build_torch_model(self.n_features, cfg).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
        loss_fn = torch.nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience = 0
        for epoch in range(1, cfg.epochs + 1):
            net.train()
            train_loss_sum = 0.0; n_seen = 0
            for xb, yb in train_dl:
                xb = xb.to(device).transpose(1, 2)   # [B, F, L]
                yb = yb.to(device)
                optim.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                if cfg.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip)
                optim.step()
                train_loss_sum += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)
            train_loss = train_loss_sum / max(n_seen, 1)
            self.train_loss_history_.append(train_loss)

            val_loss = float("nan")
            if val_dl is not None:
                net.eval()
                vsum = 0.0; vn = 0
                with torch.no_grad():
                    for xb, yb in val_dl:
                        xb = xb.to(device).transpose(1, 2); yb = yb.to(device)
                        vsum += float(loss_fn(net(xb), yb).item()) * xb.size(0); vn += xb.size(0)
                val_loss = vsum / max(vn, 1)
                self.val_loss_history_.append(val_loss)
                if val_loss < best_val - 1e-9:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
            sched.step()
            logger.info("epoch %2d/%d  train_mse=%.6f  val_mse=%.6f  patience=%d/%d",
                        epoch, cfg.epochs, train_loss, val_loss, patience, cfg.early_stopping_patience)
            if patience >= cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model_ = net.cpu()
        return self

    def predict(self, X) -> np.ndarray:
        import torch
        if self.model_ is None:
            raise RuntimeError("CNN1DRegressor must be fit before predict.")
        self.model_.eval()
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            out_chunks = []
            B = self.cfg.batch_size
            for i in range(0, len(Xt), B):
                chunk = Xt[i : i + B].transpose(1, 2)
                out_chunks.append(self.model_(chunk).cpu().numpy().reshape(-1))
        return np.concatenate(out_chunks, axis=0)

    def loss_history(self) -> dict[str, list[float]]:
        return {"train": list(self.train_loss_history_), "val": list(self.val_loss_history_)}

    def save(self, path: str | Path) -> Path:
        import torch
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), p)
        meta = {
            "n_features": self.n_features,
            "cfg": vars(self.cfg),
            "train_loss_history": self.train_loss_history_,
            "val_loss_history": self.val_loss_history_,
        }
        with p.with_suffix(".meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CNN1DRegressor":
        import torch
        p = Path(path)
        with p.with_suffix(".meta.json").open("r") as f:
            meta = json.load(f)
        cfg = CNN1DConfig(**meta["cfg"])
        m = cls(n_features=meta["n_features"], cfg=cfg)
        m.model_ = _build_torch_model(meta["n_features"], cfg)
        m.model_.load_state_dict(torch.load(p, map_location="cpu"))
        m.train_loss_history_ = meta["train_loss_history"]
        m.val_loss_history_ = meta["val_loss_history"]
        return m
