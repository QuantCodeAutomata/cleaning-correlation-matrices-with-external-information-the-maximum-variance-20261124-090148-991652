"""Out-of-sample risk-parity backtest shared by every correlation estimator.

Implements paper Sec. 4 ("Application to a universe of 47 futures
contracts"): at each week the portfolio Pi_t propto Sigma_t^{-1} C_t^{-1} 1
(the maximum-diversification portfolio of Choueifaty and Coignard, 2008) is
built from data available up to and including week t, and realised against
week t+1's return, so no information from week t+1 enters its construction.

Custom implementation (paper Sec. 4); orchestrates the EMA, MaxVar, RIE and
portfolio building blocks of ``src.ema``, ``src.maxvar``, ``src.rie`` and
``src.portfolio``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ema import (
    daily_to_weekly_returns,
    ema_correlation_sequence,
    ema_volatility,
    rescale_by_volatility,
)
from src.maxvar import maxvar_clean
from src.portfolio import max_diversification_weights, portfolio_returns
from src.rie import rie_clean


def prepare_weekly_inputs(
    returns: pd.DataFrame, vol_eta: float, corr_eta: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Weekly building blocks for the backtest.

    Returns
    -------
    vol_weekly : DataFrame (W, N)
        Daily EMA volatility (``src.ema.ema_volatility``) sampled at each
        week's last trading date (Sigma_t of paper Sec. 4).
    weekly_raw : DataFrame (W, N)
        Raw (not volatility-rescaled) weekly returns, used to realise the
        portfolio's profit and loss.
    weekly_rescaled : DataFrame (W, N)
        Volatility-rescaled weekly returns (paper Sec. 2), used only to
        estimate the correlation matrix.
    corr_sequence : list of (N, N) ndarray
        Correlation estimate using data up to and including week w (paper
        Sec. 2 EMA recursion); index w of this list aligns with row w of
        ``weekly_raw``/``weekly_rescaled``.
    """
    vol = ema_volatility(returns, vol_eta)
    rescaled = rescale_by_volatility(returns, vol)
    weekly_rescaled = daily_to_weekly_returns(rescaled)
    weekly_raw = daily_to_weekly_returns(returns)
    corr_sequence = ema_correlation_sequence(weekly_rescaled, corr_eta)
    vol_weekly = vol.loc[weekly_rescaled.index]
    return vol_weekly, weekly_raw, weekly_rescaled, corr_sequence


def run_risk_parity_backtest(
    returns: pd.DataFrame,
    groups: list[np.ndarray],
    method: str,
    vol_eta: float,
    corr_eta: float,
    effective_t: int,
    burn_in_weeks: int,
) -> pd.Series:
    """Weekly out-of-sample returns of the risk-parity portfolio (paper Sec. 4).

    Parameters
    ----------
    returns : DataFrame (T, N)
        Daily raw returns, one column per instrument (``src.config.ALL_INSTRUMENTS``
        order), as produced by ``data.data_loader.load_universe``.
    groups : list of 1-D int ndarray
        MaxVar group partition (``src.maxvar.groups_from_sizes``).
    method : {"empirical", "rie", "maxvar"}
        Correlation estimator cleaning the weekly EMA correlation matrix
        before portfolio construction.
    vol_eta : float
        EMA decay for the daily volatility estimate (paper eta).
    corr_eta : float
        EMA decay for the weekly correlation estimate (paper eta').
    effective_t : int
        Effective number of independent weekly observations behind the EMA
        correlation estimate, used as the RIE's T.
    burn_in_weeks : int
        Number of initial weeks skipped before the first portfolio is formed.

    Returns
    -------
    pd.Series
        Realised weekly portfolio returns, indexed by the date of the week
        over which the return is realised (one week after the portfolio is
        formed), named ``method``. A week is skipped (and a warning recorded
        in the returned series' ``attrs["skipped"]``) if the correlation
        matrix cannot be inverted for that method.
    """
    if method not in {"empirical", "rie", "maxvar"}:
        raise ValueError(f"unknown method: {method!r}")
    vol_weekly, weekly_raw, _, corr_sequence = prepare_weekly_inputs(returns, vol_eta, corr_eta)

    dates: list[pd.Timestamp] = []
    realised: list[float] = []
    skipped: list[str] = []
    for w in range(burn_in_weeks, len(weekly_raw) - 1):
        corr_w = corr_sequence[w]
        vol_w = vol_weekly.iloc[w].to_numpy()
        forward_return = weekly_raw.iloc[w + 1].to_numpy()
        # An instrument is active this week once its own history has started
        # (finite correlation diagonal and volatility) and its next week's
        # return is observable; instruments not yet listed (paper Sec. 4
        # real-data universe, Table 2 proxies with staggered start dates)
        # are left out of this week's portfolio rather than assigned a
        # fabricated weight, so the backtest can start as early as 1993
        # using whichever instruments are already active.
        active = np.isfinite(np.diag(corr_w)) & np.isfinite(vol_w) & np.isfinite(forward_return)
        active_idx = np.flatnonzero(active)
        local_groups = [
            np.array([pos for pos, g in enumerate(active_idx) if g in set(group)], dtype=int)
            for group in groups
        ]
        local_groups = [g for g in local_groups if g.size > 0]
        if active_idx.size < 2 or len(local_groups) < 1:
            skipped.append(str(weekly_raw.index[w]))
            continue
        corr_active = corr_w[np.ix_(active_idx, active_idx)]
        try:
            if method == "maxvar":
                corr_use = maxvar_clean(corr_active, local_groups)
            elif method == "rie":
                corr_use = rie_clean(corr_active, n_obs=effective_t)
            else:
                corr_use = corr_active
            weights_active = max_diversification_weights(vol_w[active_idx], corr_use)
        except np.linalg.LinAlgError:
            skipped.append(str(weekly_raw.index[w]))
            continue
        realised.append(portfolio_returns(weights_active, forward_return[active_idx]))
        dates.append(weekly_raw.index[w + 1])

    series = pd.Series(realised, index=pd.DatetimeIndex(dates), name=method)
    series.attrs["skipped"] = skipped
    return series
