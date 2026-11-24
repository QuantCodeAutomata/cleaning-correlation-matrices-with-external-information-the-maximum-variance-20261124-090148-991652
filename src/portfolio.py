"""Risk-parity (maximum diversification) portfolio construction.

Implements the portfolio of paper Sec. 4: Pi_t proportional to
Sigma_t^{-1} C_t^{-1} 1, the maximum diversification portfolio of
Choueifaty and Coignard (2008), where Sigma_t is the diagonal matrix of
volatilities and C_t is the (cleaned or empirical) correlation matrix.

Custom implementation (paper Eq. in Sec. 4); PyPortfolioOpt's risk-parity
solver targets equal risk contribution under a full covariance matrix
(a different, iterative objective) rather than this closed-form
diversification weighting, so the paper's explicit formula is coded
directly.
"""

from __future__ import annotations

import numpy as np


def max_diversification_weights(volatility: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Weights Pi proportional to Sigma^{-1} C^{-1} 1, normalised to sum to 1.

    Parameters
    ----------
    volatility : (n,) ndarray
        Per-instrument volatility estimate (diagonal of Sigma).
    corr : (n,) ndarray
        Correlation matrix estimate (empirical, RIE- or MaxVar-cleaned).

    Returns
    -------
    (n,) ndarray
        Portfolio weights, summing to 1.
    """
    assert np.all(volatility > 0), "volatility must be strictly positive"
    inv_vol = 1.0 / volatility
    ones = np.ones(corr.shape[0])
    raw = inv_vol * np.linalg.solve(corr, ones)
    total = raw.sum()
    assert total != 0, "portfolio weights sum to zero; cannot normalise"
    return raw / total


def portfolio_returns(weights: np.ndarray, forward_returns: np.ndarray) -> float:
    """Realised portfolio return given weights fixed before ``forward_returns``."""
    return float(weights @ forward_returns)
