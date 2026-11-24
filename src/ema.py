"""Exponential moving-average volatility and correlation estimators.

Implements paper Sec. 2 ("Methodology"): daily volatility is estimated with
an EMA of squared returns, returns are rescaled by that volatility, weekly
returns are formed from the rescaled daily returns, and the correlation
matrix is estimated with an EMA of the outer product of weekly returns.

Custom implementation (paper Eq. in Sec. 2; no library implements this
specific rescale-then-EMA-on-weekly-returns procedure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema_volatility(returns: pd.DataFrame, eta: float) -> pd.DataFrame:
    """Exponential moving-average volatility estimate.

    [v_t^j]^2 = (1 - eta) * [v_{t-1}^j]^2 + eta * [r_t^j]^2

    Parameters
    ----------
    returns : DataFrame (T, N)
        Daily returns, one column per instrument.
    eta : float
        EMA decay parameter.

    Returns
    -------
    DataFrame (T, N)
        Estimated daily volatility v_t^j (not variance), aligned with
        ``returns``. The EMA is initialised with the first squared return of
        each column (warm-up period not specified in the paper).
    """
    assert 0.0 < eta <= 1.0, "eta must be in (0, 1]"
    sq = returns.pow(2)
    var = sq.ewm(alpha=eta, adjust=False).mean()
    # ewm with adjust=False already implements v_t = (1-eta) v_{t-1} + eta x_t
    # once seeded with the first observation, matching the paper's recursion.
    return np.sqrt(var)


def rescale_by_volatility(returns: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
    """Rescaled returns r_hat_t^j = r_t^j / v_t^j (paper Sec. 2)."""
    rescaled = returns / volatility
    return rescaled.replace([np.inf, -np.inf], np.nan)


def daily_to_weekly_returns(rescaled_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rescaled daily returns into non-overlapping weekly returns.

    The paper states that returns are aggregated to a weekly frequency "to
    offset different trading hours" across exchanges, without specifying the
    aggregation rule; a simple sum of the volatility-rescaled daily returns
    (the arithmetic-return convention) is used here, grouped by ISO calendar
    week.
    """
    iso = rescaled_returns.index.isocalendar()
    weekly = rescaled_returns.groupby([iso["year"], iso["week"]]).sum(min_count=1)
    # Label each weekly row with the last trading date it contains, so the
    # resulting index is a plain, monotonically increasing DatetimeIndex.
    last_date = rescaled_returns.index.to_series().groupby([iso["year"], iso["week"]]).max()
    weekly.index = pd.DatetimeIndex(last_date.values)
    return weekly.sort_index()


def ema_correlation_sequence(weekly_returns: pd.DataFrame, eta_prime: float) -> list[np.ndarray]:
    """Sequence of EMA correlation matrices, one per weekly observation.

    C_hat_t^ij = (1 - eta') * C_hat_{t-1}^ij + eta' * r_hat_t^i * r_hat_t^j
    C_t^ij = C_hat_t^ij / sqrt(C_hat_t^ii * C_hat_t^jj)

    Parameters
    ----------
    weekly_returns : DataFrame (W, N)
        Weekly, volatility-rescaled returns (output of
        :func:`daily_to_weekly_returns`).
    eta_prime : float
        EMA decay parameter for the correlation estimate.

    Returns
    -------
    list of (N, N) ndarray
        Correlation matrix estimated using data up to and including each
        weekly observation (no look-ahead: the matrix at index ``t`` uses
        only rows ``0..t`` of ``weekly_returns``).
    """
    assert 0.0 < eta_prime <= 1.0, "eta_prime must be in (0, 1]"
    values = weekly_returns.to_numpy()
    n = values.shape[1]
    cov_hat = np.outer(values[0], values[0])
    matrices = []
    for t in range(values.shape[0]):
        if t > 0:
            cov_hat = (1 - eta_prime) * cov_hat + eta_prime * np.outer(values[t], values[t])
        d = np.sqrt(np.diag(cov_hat))
        d[d == 0] = np.nan
        corr = cov_hat / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        matrices.append(corr)
    return matrices
