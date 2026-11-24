"""Unit tests for src/ema.py on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import ema


def _synthetic_returns(n_days: int = 400, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    data = rng.standard_normal((n_days, n_assets)) * 0.01
    return pd.DataFrame(data, index=dates, columns=[f"a{i}" for i in range(n_assets)])


def test_ema_volatility_matches_recursion() -> None:
    returns = _synthetic_returns(50, 2)
    eta = 0.2
    vol = ema.ema_volatility(returns, eta)
    values = returns.to_numpy()
    var = np.zeros_like(values)
    var[0] = values[0] ** 2
    for t in range(1, len(values)):
        var[t] = (1 - eta) * var[t - 1] + eta * values[t] ** 2
    np.testing.assert_allclose(vol.to_numpy(), np.sqrt(var), rtol=1e-10)


def test_ema_volatility_rejects_bad_eta() -> None:
    returns = _synthetic_returns(10, 2)
    with pytest.raises(AssertionError):
        ema.ema_volatility(returns, 0.0)
    with pytest.raises(AssertionError):
        ema.ema_volatility(returns, 1.5)


def test_rescale_by_volatility_handles_zero_vol() -> None:
    returns = pd.DataFrame({"a": [0.0, 0.01]})
    vol = pd.DataFrame({"a": [0.0, 0.01]})
    rescaled = ema.rescale_by_volatility(returns, vol)
    assert np.isnan(rescaled["a"].iloc[0])
    assert rescaled["a"].iloc[1] == pytest.approx(1.0)


def test_daily_to_weekly_returns_reduces_rows() -> None:
    returns = _synthetic_returns(250, 3)
    weekly = ema.daily_to_weekly_returns(returns)
    assert len(weekly) < len(returns)
    assert weekly.index.is_monotonic_increasing
    # Roughly 5 trading days per week.
    assert 45 <= len(weekly) <= 55


def test_ema_correlation_sequence_no_lookahead_and_valid_correlation() -> None:
    returns = _synthetic_returns(60, 3)
    weekly = ema.daily_to_weekly_returns(returns)
    matrices = ema.ema_correlation_sequence(weekly, eta_prime=0.1)
    assert len(matrices) == len(weekly)
    for corr in matrices:
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr, corr.T)
        eigvals = np.linalg.eigvalsh(corr)
        assert eigvals.min() > -1e-8  # positive semi-definite

    # No look-ahead: recomputing with only the first half of the data must
    # reproduce exactly the first half of the sequence computed on the full data.
    half = len(weekly) // 2
    matrices_half = ema.ema_correlation_sequence(weekly.iloc[:half], eta_prime=0.1)
    for full, partial in zip(matrices[:half], matrices_half):
        np.testing.assert_allclose(full, partial, rtol=1e-10)


def test_ema_correlation_sequence_handles_staggered_entry() -> None:
    """An instrument listed later (paper Sec. 4 real-data universe, where
    several Table 2 proxies start trading only years after the others) must
    not contaminate the other entries before it has any data, and must join
    the recursion cleanly once its own history begins."""
    returns = _synthetic_returns(80, 3)
    weekly = ema.daily_to_weekly_returns(returns)
    late_start = len(weekly) // 2
    staggered = weekly.copy()
    staggered.iloc[:late_start, 2] = np.nan

    matrices = ema.ema_correlation_sequence(staggered, eta_prime=0.1)
    for t in range(late_start):
        assert np.isnan(matrices[t][2, :]).all()
        assert np.isnan(matrices[t][:, 2]).all()
        active = matrices[t][:2, :2]
        assert np.allclose(np.diag(active), 1.0)

    for t in range(late_start, len(weekly)):
        corr = matrices[t]
        assert not np.isnan(corr).any()
        assert np.allclose(np.diag(corr), 1.0)
        eigvals = np.linalg.eigvalsh(corr)
        assert eigvals.min() > -1e-8

    # The two always-active instruments must match a sequence computed with
    # no missing data at all: the third instrument's absence must not alter
    # their own pairwise recursion.
    dense_matrices = ema.ema_correlation_sequence(weekly, eta_prime=0.1)
    for t in range(len(weekly)):
        np.testing.assert_allclose(matrices[t][:2, :2], dense_matrices[t][:2, :2], rtol=1e-10)
