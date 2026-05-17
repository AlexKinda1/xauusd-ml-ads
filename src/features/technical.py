"""Technical indicators and price-derived features.

Hand-written pure-pandas / numpy implementations. Two reasons over pandas-ta:

  1. Every indicator is *causal by construction* — values at index ``t`` only
     depend on prices at indices ``<= t``. This is verified by
     ``tests/test_anti_leakage.py`` and is the cornerstone of the project's
     scientific rigour.
  2. We avoid pinning to a specific pandas-ta release whose ``numpy``
     compatibility window has historically been tight.

All public functions accept an OHLC DataFrame and return either a Series
or a small DataFrame indexed on the same timestamps. The *final* shift
that enforces "features at t use only data <= t-1" is applied by
``build_technical_features`` — individual indicators are left in their
natural (close-of-bar-t) timing so unit tests can reason about them
directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns & rolling stats
# ---------------------------------------------------------------------------


def log_returns(close: pd.Series, lag: int = 1) -> pd.Series:
    """``log(C_t / C_{t-lag})`` — causal."""
    return np.log(close).diff(lag).rename(f"logret_{lag}")


def realized_volatility(close: pd.Series, window: int = 24) -> pd.Series:
    """Rolling std of 1-bar log-returns over ``window`` bars."""
    r = np.log(close).diff()
    return r.rolling(window, min_periods=window).std().rename(f"rv_{window}")


def zscore(close: pd.Series, window: int = 24) -> pd.Series:
    """Rolling z-score of close vs its ``window`` SMA / STD."""
    mu = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    return ((close - mu) / sd).rename(f"zscore_{window}")


# ---------------------------------------------------------------------------
# Momentum oscillators
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Returns NaN for the first ``period`` rows.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.rename(f"rsi_{period}")


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, histogram (Appel 1979). All EMAs are causal."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame(
        {"macd_line": line, "macd_signal": sig, "macd_hist": hist}
    )


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change : ``(C_t / C_{t-period} - 1) * 100``."""
    return (close.pct_change(period) * 100).rename(f"roc_{period}")


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R in [-100, 0]."""
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return (-100.0 * (hh - close) / (hh - ll)).rename(f"willr_{period}")


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator %K and %D."""
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    pk = 100.0 * (close - ll) / (hh - ll)
    pd_ = pk.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"stoch_k": pk, "stoch_d": pd_})


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=period).mean()
    return ((tp - sma) / (0.015 * mad)).rename(f"cci_{period}")


# ---------------------------------------------------------------------------
# Volatility / bands
# ---------------------------------------------------------------------------


def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: upper, lower, %b, band-width."""
    sma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std()
    upper = sma + n_std * sd
    lower = sma - n_std * sd
    pct_b = (close - lower) / (upper - lower)
    bw = (upper - lower) / sma
    return pd.DataFrame(
        {"bb_upper": upper, "bb_lower": lower, "bb_pctb": pct_b, "bb_bw": bw}
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().rename(
        f"atr_{period}"
    )


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().rename(
        f"adx_{period}"
    )


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------


def build_technical_features(
    ohlcv: pd.DataFrame,
    *,
    feature_lag: int = 1,
) -> pd.DataFrame:
    """Compute the full bank of technical features and shift them by ``feature_lag``.

    The shift enforces the project's anti-leakage convention: features at
    index ``t`` describe market state up to and including bar ``t - feature_lag``.
    With ``feature_lag = 1`` (default) features at ``t`` use only bars ``<= t-1``.

    Args:
        ohlcv: DataFrame with columns ``open, high, low, close, volume``.
        feature_lag: How many bars to shift the feature frame forward
            (i.e. into the future) so that row ``t`` uses only past bars.

    Returns:
        Feature DataFrame indexed on ``ohlcv.index`` with ``~30`` columns.
    """
    o, h, ll, c = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]

    pieces: list[pd.Series | pd.DataFrame] = [
        log_returns(c, 1),
        log_returns(c, 4),
        log_returns(c, 24),
        log_returns(c, 168),
        realized_volatility(c, 24),
        realized_volatility(c, 168),
        zscore(c, 24),
        zscore(c, 168),
        rsi(c, 14),
        macd(c),
        roc(c, 10),
        williams_r(h, ll, c, 14),
        stochastic(h, ll, c),
        cci(h, ll, c, 20),
        bollinger(c, 20, 2.0),
        atr(h, ll, c, 14),
        adx(h, ll, c, 14),
    ]
    feats = pd.concat(pieces, axis=1)
    if feature_lag > 0:
        feats = feats.shift(feature_lag)
    return feats
