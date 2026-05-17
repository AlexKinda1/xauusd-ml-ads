"""Phase-3 entry point: build splits, fit scaler, materialise per-family datasets.

Outputs to ``data/processed/`` :

- ``splits/{train,val,test}_tabular.parquet`` — tree-model inputs (X + target).
- ``splits/{train,val,test}_sequences.npz`` — DL-model inputs (X, y, end_ts).
- ``scaler.pkl`` — fitted scaler for inference reproducibility.
- ``reports/tables/phase3_summary.json`` — shapes and per-split metadata.

Usage::

    poetry run python scripts/03_build_splits.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import scaling, sequences, splits
from src.utils.config import PROJECT_ROOT, load_training_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

FEATURES_PATH = "data/processed/features_targets.parquet"
OUT_DIR = "data/processed/splits"
SCALER_PATH = "data/processed/scaler.pkl"


def main() -> None:
    set_global_seed(42)
    cfg = load_training_config()
    task = cfg["task"]
    split_cfg = cfg["splits"]

    horizon = int(task["horizon"])
    lookback = int(task["lookback"])
    embargo = int(split_cfg["embargo_bars"])

    df = pd.read_parquet(PROJECT_ROOT / FEATURES_PATH)
    feature_cols = [
        c for c in df.columns
        if c not in {"open", "high", "low", "close", "volume",
                     f"y_reg_h{horizon}", f"y_clf_h{horizon}", "y_clf_threshold"}
    ]
    logger.info("Loaded %d rows, %d features", len(df), len(feature_cols))

    # 1. Splits
    sp = splits.chronological_split(
        df.index,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        embargo_bars=embargo,
    )
    for s in sp.values():
        logger.info("%s", s)

    train_df = splits.apply_split(df, sp["train"])
    val_df = splits.apply_split(df, sp["val"])
    test_df = splits.apply_split(df, sp["test"])

    # 2. Scaler — fit on train only
    scaler = scaling.TrainOnlyScaler.fit(train_df, feature_cols=feature_cols)
    scaler_path = PROJECT_ROOT / SCALER_PATH
    scaler.save(scaler_path)
    logger.info("Saved scaler -> %s", scaler_path)

    # 3. Tabular outputs — NaN preserved for XGBoost; DL gets fillna=0 sequences
    out_dir = PROJECT_ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tabular_shapes: dict[str, tuple[int, int]] = {}
    sequence_shapes: dict[str, tuple[int, int, int]] = {}

    target_reg = f"y_reg_h{horizon}"
    target_clf = f"y_clf_h{horizon}"

    for name, segment_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # Scale features (NaN preserved for tree models)
        X_scaled_tree = scaler.transform(segment_df[feature_cols], fillna=None)
        # Carry targets and a few un-scaled columns useful for diagnostics
        tab = X_scaled_tree.copy()
        tab[target_reg] = segment_df[target_reg].values
        tab[target_clf] = segment_df[target_clf].values
        tab["close"] = segment_df["close"].values
        tab_path = out_dir / f"{name}_tabular.parquet"
        tab.to_parquet(tab_path, engine="pyarrow", compression="snappy")
        tabular_shapes[name] = (len(tab), len(feature_cols))
        logger.info("Wrote %s (%d rows) -> %s", name, len(tab), tab_path)

        # Scale features (NaN -> 0 for DL); keep full pre-split context so the
        # first window of each segment uses the lookback bars preceding it
        # from the same scaler.
        X_scaled_dl = scaler.transform(df.loc[segment_df.index, feature_cols], fillna=0.0)
        # We need the L-1 bars BEFORE the segment to build the first window.
        # Reuse df with scaled values to compute sequences end-anchored inside
        # the segment.
        full_scaled = scaler.transform(df[feature_cols], fillna=0.0)
        full_scaled[target_reg] = df[target_reg].values
        full_scaled[target_clf] = df[target_clf].values

        # Slice the index to: [first_segment_row - lookback + 1, last_segment_row]
        first_pos = df.index.get_loc(segment_df.index[0])
        last_pos = df.index.get_loc(segment_df.index[-1])
        window_start = max(first_pos - lookback + 1, 0)
        sub = full_scaled.iloc[window_start : last_pos + 1]

        X_seq, y_reg_seq, end_ts = sequences.build_sequences(
            sub, feature_cols, target_reg, lookback=lookback
        )
        # Build classification target on the same windows.
        _, y_clf_seq, _ = sequences.build_sequences(
            sub, feature_cols, target_clf, lookback=lookback
        )
        # Restrict to windows whose END falls inside the segment.
        in_segment = (end_ts >= segment_df.index[0]) & (end_ts <= segment_df.index[-1])
        X_seq = X_seq[in_segment]
        y_reg_seq = y_reg_seq[in_segment]
        y_clf_seq = y_clf_seq[in_segment]
        end_ts = end_ts[in_segment]

        seq_path = out_dir / f"{name}_sequences.npz"
        np.savez_compressed(
            seq_path,
            X=X_seq.astype("float32"),
            y_reg=y_reg_seq.astype("float32"),
            y_clf=y_clf_seq.astype("float32"),
            end_ts=end_ts.tz_convert("UTC").tz_localize(None).astype("datetime64[ns]").values,
            feature_cols=np.array(feature_cols, dtype=object),
        )
        sequence_shapes[name] = tuple(X_seq.shape)
        logger.info("Wrote %s sequences shape=%s -> %s", name, X_seq.shape, seq_path)

    # 4. Walk-forward folds summary
    wf_summary = []
    for train_fold, val_fold in splits.walk_forward_expanding(
        df.index, n_folds=int(cfg["walk_forward"]["n_folds"]), embargo_bars=embargo,
    ):
        wf_summary.append({
            "fold": val_fold.name,
            "train_rows": train_fold.n_rows,
            "val_rows": val_fold.n_rows,
            "train_end": str(train_fold.end_ts),
            "val_start": str(val_fold.start_ts),
            "val_end": str(val_fold.end_ts),
        })

    summary = {
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "splits": {
            name: {
                "rows": s.n_rows,
                "start": str(s.start_ts),
                "end": str(s.end_ts),
                "start_idx": s.start_idx,
                "end_idx": s.end_idx,
            } for name, s in sp.items()
        },
        "tabular_shapes": {k: list(v) for k, v in tabular_shapes.items()},
        "sequence_shapes": {k: list(v) for k, v in sequence_shapes.items()},
        "walk_forward": wf_summary,
        "scaler_path": str(scaler_path.relative_to(PROJECT_ROOT)),
    }
    out_summary = PROJECT_ROOT / "reports/tables/phase3_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Phase 3 complete. Summary -> %s", out_summary)


if __name__ == "__main__":
    main()
