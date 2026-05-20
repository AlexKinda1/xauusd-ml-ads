"""Statistical tests for rigorous model comparison.

- **Diebold-Mariano** : compares the predictive accuracy of two forecast
  series under a chosen loss (squared / absolute). Accounts for forecast
  horizon autocorrelation via a HAC (Newey-West) variance estimator.
- **McNemar** : paired test for two classifiers on the same samples.
- **Binomial directional test** : is directional accuracy significantly
  above 0.5? Exact / normal-approx two-sided test.
- **Bootstrap Sharpe CI** : block-bootstrap confidence interval for an
  annualised Sharpe ratio (handles autocorrelation in overlapping returns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    statistic: float
    p_value: float
    detail: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"TestResult(stat={self.statistic:.4f}, p={self.p_value:.4g}) {self.detail}"


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


def diebold_mariano(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    *,
    horizon: int = 1,
    loss: str = "squared",
) -> TestResult:
    """Diebold-Mariano test of equal predictive accuracy.

    H0: the two forecasts have equal expected loss.
    Negative statistic => ``pred1`` is more accurate (lower loss) than ``pred2``.

    Uses a Newey-West HAC variance with ``horizon - 1`` lags, the standard
    correction for ``h``-step-ahead overlapping forecasts.
    """
    mask = ~(np.isnan(y_true) | np.isnan(pred1) | np.isnan(pred2))
    yt, p1, p2 = y_true[mask], pred1[mask], pred2[mask]
    if loss == "squared":
        e1 = (yt - p1) ** 2
        e2 = (yt - p2) ** 2
    elif loss == "absolute":
        e1 = np.abs(yt - p1)
        e2 = np.abs(yt - p2)
    else:
        raise ValueError(f"Unknown loss: {loss!r}")

    d = e1 - e2
    n = len(d)
    d_bar = d.mean()

    # Newey-West long-run variance with (horizon-1) lags.
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0
    for lag in range(1, horizon):
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        weight = 1.0 - lag / horizon
        lrv += 2.0 * weight * cov
    lrv = max(lrv, 1e-18)

    dm_stat = d_bar / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold small-sample correction
    k = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_stat *= k
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=n - 1))
    direction = "pred1 better" if d_bar < 0 else "pred2 better"
    return TestResult(float(dm_stat), float(p_value), f"{direction} (loss={loss})")


# ---------------------------------------------------------------------------
# McNemar (classification)
# ---------------------------------------------------------------------------


def mcnemar(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray,
            *, continuity_correction: bool = True) -> TestResult:
    """McNemar test for two classifiers on the same labelled samples."""
    mask = ~(np.isnan(y_true.astype("float64")))
    yt = y_true[mask].astype(int)
    c1 = pred1[mask].astype(int) == yt
    c2 = pred2[mask].astype(int) == yt
    # b: pred1 wrong, pred2 right ; c: pred1 right, pred2 wrong
    b = int(np.sum(~c1 & c2))
    c = int(np.sum(c1 & ~c2))
    if b + c == 0:
        return TestResult(0.0, 1.0, "identical predictions")
    cc = 1.0 if continuity_correction else 0.0
    chi2 = (abs(b - c) - cc) ** 2 / (b + c)
    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)
    return TestResult(float(chi2), float(p_value), f"b={b}, c={c}")


# ---------------------------------------------------------------------------
# Directional accuracy significance
# ---------------------------------------------------------------------------


def binomial_directional_test(n_correct: int, n_total: int, p0: float = 0.5) -> TestResult:
    """Two-sided test that directional accuracy != ``p0`` (default 0.5).

    Uses the normal approximation (valid for the large n we have, ~15k).
    """
    if n_total == 0:
        return TestResult(float("nan"), float("nan"), "no samples")
    p_hat = n_correct / n_total
    se = np.sqrt(p0 * (1 - p0) / n_total)
    z = (p_hat - p0) / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return TestResult(float(z), float(p_value), f"acc={p_hat:.4f} ({n_correct}/{n_total})")


# ---------------------------------------------------------------------------
# Bootstrap Sharpe CI
# ---------------------------------------------------------------------------


def bootstrap_sharpe_ci(
    pnl: np.ndarray,
    *,
    periods_per_year: float,
    n_boot: int = 2000,
    block_size: int = 24,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Block-bootstrap CI for an annualised Sharpe ratio.

    Block bootstrap (moving blocks) preserves autocorrelation present in
    overlapping h-step returns.

    Returns ``(sharpe_point, lo, hi)``.
    """
    pnl = pnl[~np.isnan(pnl)]
    n = len(pnl)
    if n < block_size * 2 or pnl.std() == 0:
        s = float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year)) if pnl.std() > 0 else 0.0
        return s, float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    point = float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))
    n_blocks = int(np.ceil(n / block_size))
    sharpes = np.empty(n_boot)
    max_start = n - block_size
    for b in range(n_boot):
        starts = rng.randint(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([pnl[s:s + block_size] for s in starts])[:n]
        sd = sample.std()
        sharpes[b] = sample.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(sharpes, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)
