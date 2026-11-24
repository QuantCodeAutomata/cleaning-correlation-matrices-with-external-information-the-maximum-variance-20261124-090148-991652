"""Unit tests for src/rie.py (RIE of Bun, Bouchaud, Potters) on synthetic data."""

from __future__ import annotations

import numpy as np

from src import rie


def _mp_sample_correlation(n: int, t: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t, n))
    corr = np.corrcoef(x, rowvar=False)
    return corr


def test_rie_clean_output_is_valid_correlation_matrix() -> None:
    corr = _mp_sample_correlation(20, 100, seed=0)
    cleaned = rie.rie_clean(corr, n_obs=100)
    np.testing.assert_allclose(np.diag(cleaned), 1.0, atol=1e-8)
    np.testing.assert_allclose(cleaned, cleaned.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(cleaned)
    assert eigvals.min() > -1e-6


def test_rie_clean_reduces_distance_for_identity_truth() -> None:
    """Under a true identity correlation (no structure), the empirical matrix
    is pure noise; RIE shrinkage should move the estimate closer to the
    identity than the raw empirical correlation (paper Sec. 3 / Physics
    Reports Sec. 5-6 property of the oracle eigenvalue estimator)."""
    n, t = 30, 150
    truth = np.eye(n)
    distances_empirical = []
    distances_cleaned = []
    for seed in range(10):
        corr = _mp_sample_correlation(n, t, seed)
        cleaned = rie.rie_clean(corr, n_obs=t)
        distances_empirical.append(np.linalg.norm(corr - truth, ord="fro"))
        distances_cleaned.append(np.linalg.norm(cleaned - truth, ord="fro"))
    assert np.mean(distances_cleaned) < np.mean(distances_empirical)


def test_rie_clean_identity_input_is_near_fixed_point() -> None:
    cleaned = rie.rie_clean(np.eye(10), n_obs=200)
    np.testing.assert_allclose(cleaned, np.eye(10), atol=1e-3)
