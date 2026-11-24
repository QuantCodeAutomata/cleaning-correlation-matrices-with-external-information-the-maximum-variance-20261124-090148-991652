"""Unit tests for src/metrics.py and src/portfolio.py on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics, portfolio


def test_frobenius_distance_zero_for_identical_matrices() -> None:
    m = np.eye(5)
    assert metrics.frobenius_distance(m, m) == pytest.approx(0.0, abs=1e-12)


def test_frobenius_distance_matches_manual_formula() -> None:
    a = np.array([[1.0, 0.5], [0.5, 1.0]])
    b = np.array([[1.0, 0.2], [0.2, 1.0]])
    expected = np.sqrt(np.sum((a - b) ** 2))
    assert metrics.frobenius_distance(a, b) == pytest.approx(expected)


def test_annualized_sharpe_ratio_sign_and_scale() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
    sharpe = metrics.annualized_sharpe_ratio(returns, periods_per_year=252)
    manual = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe == pytest.approx(manual)
    negative_returns = -returns
    assert metrics.annualized_sharpe_ratio(negative_returns, 252) == pytest.approx(-sharpe)


def test_annualized_sharpe_ratio_rejects_constant_series() -> None:
    with pytest.raises(AssertionError):
        metrics.annualized_sharpe_ratio(pd.Series([0.01, 0.01, 0.01]), 252)


def test_max_diversification_weights_sum_to_one_and_equal_vol_case() -> None:
    vol = np.array([0.1, 0.2, 0.15])
    corr = np.eye(3)
    weights = portfolio.max_diversification_weights(vol, corr)
    assert weights.sum() == pytest.approx(1.0)
    # With zero correlation, weights are proportional to 1/vol.
    expected = (1.0 / vol) / (1.0 / vol).sum()
    np.testing.assert_allclose(weights, expected)


def test_max_diversification_weights_rejects_nonpositive_vol() -> None:
    with pytest.raises(AssertionError):
        portfolio.max_diversification_weights(np.array([0.1, -0.1]), np.eye(2))


def test_portfolio_returns_matches_dot_product() -> None:
    weights = np.array([0.5, 0.5])
    forward = np.array([0.02, -0.01])
    assert portfolio.portfolio_returns(weights, forward) == pytest.approx(0.005)
