"""Walk-forward training framework.

Standard finance-quant validation : we refit the model periodically on the
most recent data and predict only the immediately following segment. This
keeps the model in the same regime as the prediction window — the failure
mode that broke every static train→test model in Phase 4.

Workflow per fold ``k``::

    train  : [0, fold_start_k - embargo)
    fold   : [fold_start_k, fold_start_k + fold_size)

The function refits the model from scratch at each fold by calling a
user-supplied factory; this avoids any global state leaking between folds.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

from src.models.base import ModelBase
from src.utils.logging import get_logger

logger = get_logger(__name__)


def walk_forward_predictions(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    model_factory: Callable[[], ModelBase],
    initial_train_end_idx: int,
    fold_size: int,
    embargo: int,
    n_folds: int | None = None,
    extra_cols: list[str] | None = None,
    fit_kwargs_factory: Callable[[pd.DataFrame, pd.DataFrame], dict] | None = None,
    train_window: int | None = None,
) -> pd.DataFrame:
    """Generate walk-forward predictions over a feature DataFrame.

    Args:
        df: Full feature DataFrame, chronologically sorted, indexed by datetime.
        feature_cols: Columns to use as inputs.
        target_col: Column to predict.
        model_factory: Callable returning a fresh :class:`ModelBase` per fold.
        initial_train_end_idx: Integer index of the first row OUTSIDE the
            initial training window (i.e. where folding starts).
        fold_size: Number of rows per fold.
        embargo: Number of rows to skip between train end and fold start.
        n_folds: Cap on number of folds (default = as many as fit).
        extra_cols: Extra columns from ``df`` to carry into the output (e.g.
            ``"close"`` for downstream Sharpe computation).
        fit_kwargs_factory: Optional ``(train_df, fold_df) -> kwargs`` for
            ``model.fit`` (e.g. to pass an X_val constructed from train tail).
        train_window: If ``None`` (default), training is **expanding** —
            every fold sees all data from index 0 up to its embargoed train end.
            If an integer, training is **sliding** — each fold uses only the
            last ``train_window`` rows ending at ``fold_start - embargo``.

    Returns:
        DataFrame indexed on the union of all fold windows with columns
        ``y_true``, ``y_pred`` and optionally any column from ``extra_cols``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("walk_forward_predictions requires a DatetimeIndex.")
    if initial_train_end_idx <= 0 or initial_train_end_idx >= len(df):
        raise ValueError(f"Bad initial_train_end_idx: {initial_train_end_idx} / {len(df)}")

    extra_cols = extra_cols or []
    out_chunks: list[pd.DataFrame] = []

    fold_idx = 0
    fold_start = initial_train_end_idx
    while fold_start + fold_size <= len(df):
        if n_folds is not None and fold_idx >= n_folds:
            break

        fold_end = fold_start + fold_size
        train_end = max(0, fold_start - embargo)
        if train_end <= 0:
            logger.warning("Skipping fold %d — no train data left after embargo.", fold_idx)
            fold_start = fold_end
            fold_idx += 1
            continue
        train_start = 0 if train_window is None else max(0, train_end - train_window)

        train_df = df.iloc[train_start:train_end]
        fold_df = df.iloc[fold_start:fold_end]

        # Drop rows with NaN target — usually the tail or warm-up.
        train_clean = train_df.dropna(subset=[target_col])
        if len(train_clean) < 100:
            logger.warning("Skipping fold %d — train too small (%d rows).", fold_idx, len(train_clean))
            fold_start = fold_end
            fold_idx += 1
            continue

        window_kind = "expanding" if train_window is None else f"sliding[{train_window}]"
        logger.info(
            "Fold %d (%s) | train=[%d:%d] (%d rows) | fold=[%d:%d] (%d rows) | %s -> %s",
            fold_idx, window_kind, train_start, train_end, len(train_clean),
            fold_start, fold_end, fold_df.shape[0],
            train_clean.index[-1], fold_df.index[0],
        )

        model = model_factory()
        fit_kwargs = fit_kwargs_factory(train_clean, fold_df) if fit_kwargs_factory else {}
        model.fit(train_clean, train_clean[target_col].values, **fit_kwargs)
        y_pred = model.predict(fold_df)

        chunk = pd.DataFrame(
            {"y_true": fold_df[target_col].values, "y_pred": y_pred},
            index=fold_df.index,
        )
        for c in extra_cols:
            if c in fold_df.columns:
                chunk[c] = fold_df[c].values
        chunk["fold"] = fold_idx
        out_chunks.append(chunk)

        fold_start = fold_end
        fold_idx += 1

    if not out_chunks:
        raise RuntimeError("Walk-forward produced no folds. Check parameters.")
    return pd.concat(out_chunks, axis=0)


def fold_boundaries(
    df: pd.DataFrame,
    *,
    initial_train_end_idx: int,
    fold_size: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(fold_idx, start, end)`` integer triples for inspection / planning."""
    fold_idx = 0
    fold_start = initial_train_end_idx
    while fold_start + fold_size <= len(df):
        yield fold_idx, fold_start, fold_start + fold_size
        fold_idx += 1
        fold_start += fold_size
