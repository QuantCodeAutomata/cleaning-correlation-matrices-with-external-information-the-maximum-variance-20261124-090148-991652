"""Evaluation metrics shared by both experiments.

Custom implementations of the paper's Frobenius distance (Sec. 3) and the
annualised Sharpe ratio (Sec. 4); quantstats computes Sharpe ratios from a
return series with its own annualisation and risk-free handling, but the
paper's exact convention (arithmetic mean/ std of returns, scaled by
sqrt(252)) is reproduced directly to match Sec. 4 and avoid hidden defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def frobenius_distance(c_hat: np.ndarray, c0: np.ndarray) -> float:
    """d(C_hat, C0) = sqrt(Tr[(C_hat - C0)^T (C_hat - C0)]) (paper Sec. 3)."""
    diff = c_hat - c0
    return float(np.sqrt(np.trace(diff.T @ diff)))


def annualized_sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Sharpe ratio = mean(return) / std(return) * sqrt(periods_per_year).

    ``returns`` holds one observation per rebalancing period (e.g. one
    return per week for a weekly-rebalanced portfolio); ``periods_per_year``
    is the number of such periods in a year.
    """
    returns = returns.dropna()
    assert len(returns) > 1, "need at least two return observations"
    mean = returns.mean()
    std = returns.std(ddof=1)
    assert std > 0, "zero-variance return series; Sharpe ratio undefined"
    return float(mean / std * np.sqrt(periods_per_year))
