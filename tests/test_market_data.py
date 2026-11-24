"""Tests on a reduced real-data slice (paper Sec. 4 universe, three instruments)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ema import daily_to_weekly_returns, ema_correlation_sequence, ema_volatility, rescale_by_volatility


@pytest.mark.market_data
def test_ema_volatility_on_real_slice_is_nonnegative_and_no_lookahead(market_data_returns):
    vol = ema_volatility(market_data_returns, eta=1.0 / 100)
    assert (vol.dropna() >= 0).all().all()
    # No look-ahead: volatility at t must not depend on returns after t.
    half = len(market_data_returns) // 2
    vol_prefix = ema_volatility(market_data_returns.iloc[:half], eta=1.0 / 100)
    np.testing.assert_allclose(vol.iloc[:half].to_numpy(), vol_prefix.to_numpy(), rtol=1e-10)


@pytest.mark.market_data
def test_weekly_correlation_sequence_on_real_slice_is_valid_correlation(market_data_returns):
    vol = ema_volatility(market_data_returns, eta=1.0 / 100)
    rescaled = rescale_by_volatility(market_data_returns, vol)
    weekly = daily_to_weekly_returns(rescaled)
    corr_sequence = ema_correlation_sequence(weekly, eta_prime=1.0 / 52)
    last = corr_sequence[-1]
    finite = np.isfinite(last)
    assert np.allclose(np.diag(last)[np.isfinite(np.diag(last))], 1.0)
    assert np.all(last[finite] <= 1.0 + 1e-8) and np.all(last[finite] >= -1.0 - 1e-8)
