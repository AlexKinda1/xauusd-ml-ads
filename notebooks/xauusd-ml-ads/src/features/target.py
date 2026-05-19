"""Regression and classification targets.

Regression target
    ``y_reg(t) = log(P_{t+h} / P_t)``

    This is the cumulative log-return between the bar that closes at ``t``
    (which is "now" in the prediction scenario) and the bar that closes at
    ``t + h``. We *do* use ``P_t`` here — it is the **anchor**, not a feature.
    All inputs to the model live in the feature frame, which is shifted by
    ``feature_lag = 1`` independently.

Classification target (ternary)
    ``y_clf(t) = sign(y_reg(t))`` whenever ``|y_reg(t)| > threshold(t)``,
    else 0 (neutral zone).

    Threshold uses **past** realised volatility only:

        threshold(t) = factor * std(log_returns_{t-window+1..t-1}) * sqrt(h)

    This is critical: the threshold must be knowable at time ``t``,
    otherwise the class boundary itself leaks future information.
    Using strictly-past returns (``t-1`` and earlier) makes ``y_clf``
    well-defined under the project's strict anti-leakage rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_target(close: pd.Series, horizon: int = 24) -> pd.Series:
    """``log(P_{t+h} / P_t)`` — forward log-return.

    Rows where ``t + h`` extends past the end of the series are NaN.
    """
    fwd = np.log(close).shift(-horizon) - np.log(close)
    return fwd.rename(f"y_reg_h{horizon}")


def realized_vol_past(close: pd.Series, window: int = 24, lag: int = 1) -> pd.Series:
    """Rolling std of 1-bar log-returns over the *past* ``window`` bars.

    ``lag`` shifts the result forward so that the value at row ``t`` only
    depends on returns up to and including ``t - lag``. With ``lag = 1``
    this is "vol of returns from ``t - window`` to ``t - 1``".
    """
    r = np.log(close).diff()
    vol = r.rolling(window, min_periods=window).std()
    if lag > 0:
        vol = vol.shift(lag)
    return vol.rename(f"rv_past_{window}")


def classification_target(
    y_reg: pd.Series,
    close: pd.Series,
    *,
    horizon: int = 24,
    vol_window: int = 24,
    threshold_factor: float = 0.5,
) -> pd.Series:
    """Ternary direction with adaptive neutral zone.

    Args:
        y_reg: Regression target series ``y_reg(t) = log(P_{t+h}/P_t)``.
        close: Aligned close-price series (used for the past volatility
            estimate that defines the neutral threshold).
        horizon: Same ``h`` as for ``y_reg``; used to scale the per-bar
            vol estimate to a per-horizon scale.
        vol_window: Window length for the rolling realised volatility.
        threshold_factor: Multiplier on the scaled vol — wider factor =
            more neutral labels, fewer extreme labels.

    Returns:
        Integer Series in ``{-1, 0, 1}`` with the same index as ``y_reg``.
        NaN rows are kept as NaN (typically the tail where ``y_reg`` is
        undefined).
    """
    sigma_1bar = realized_vol_past(close, vol_window, lag=1)
    threshold = threshold_factor * sigma_1bar * np.sqrt(horizon)

    mask = y_reg.notna() & threshold.notna()
    label = pd.Series(np.nan, index=y_reg.index, dtype="float64")
    label.loc[mask & (y_reg > threshold)] = 1.0
    label.loc[mask & (y_reg < -threshold)] = -1.0
    label.loc[mask & (y_reg.abs() <= threshold)] = 0.0
    return label.rename(f"y_clf_h{horizon}")


def build_targets(
    close: pd.Series,
    *,
    horizon: int = 24,
    vol_window: int = 24,
    threshold_factor: float = 0.5,
) -> pd.DataFrame:
    """Combine regression + classification targets into a single DataFrame.

    Also returns the threshold series for diagnostic plots.
    """
    y_reg = regression_target(close, horizon)
    y_clf = classification_target(
        y_reg, close,
        horizon=horizon, vol_window=vol_window, threshold_factor=threshold_factor,
    )
    sigma = realized_vol_past(close, vol_window, lag=1)
    threshold = (threshold_factor * sigma * np.sqrt(horizon)).rename("y_clf_threshold")
    return pd.concat([y_reg, y_clf, threshold], axis=1)
