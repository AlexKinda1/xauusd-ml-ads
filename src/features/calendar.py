"""Calendar / seasonality features.

All features here are deterministic functions of the timestamp itself and
contain no future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sin_cos(values: pd.Series, period: int) -> pd.DataFrame:
    """Encode a cyclic integer feature with sin/cos to avoid the discontinuity
    at the period boundary (e.g. hour 23 → 0)."""
    angle = 2.0 * np.pi * values / period
    return pd.DataFrame(
        {f"{values.name}_sin": np.sin(angle), f"{values.name}_cos": np.cos(angle)}
    )


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build calendar features from a tz-aware H1 index.

    Returns:
        DataFrame indexed on ``index`` with columns:
        ``hour_sin/cos``, ``dow_sin/cos``, ``month_sin/cos``,
        ``is_session_asia/eu/us`` (overlapping trading sessions in UTC).
    """
    if index.tz is None:
        raise ValueError("calendar_features requires a tz-aware index.")

    df = pd.DataFrame(index=index)
    hour = pd.Series(index.hour, index=index, name="hour")
    dow = pd.Series(index.dayofweek, index=index, name="dow")
    month = pd.Series(index.month - 1, index=index, name="month")  # 0..11

    df = pd.concat([df, _sin_cos(hour, 24), _sin_cos(dow, 7), _sin_cos(month, 12)], axis=1)

    # Trading-session indicators (UTC):
    # - Asia    : 23:00 → 08:00
    # - Europe  : 07:00 → 16:00
    # - US      : 13:00 → 22:00
    h = index.hour
    df["is_session_asia"] = ((h >= 23) | (h < 8)).astype("int8")
    df["is_session_eu"] = ((h >= 7) & (h < 16)).astype("int8")
    df["is_session_us"] = ((h >= 13) & (h < 22)).astype("int8")
    return df
