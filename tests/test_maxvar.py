"""Unit tests for src/maxvar.py (paper Sec. 3) on synthetic correlation matrices."""

from __future__ import annotations

import numpy as np
import pytest

from src import maxvar


def _random_correlation(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n * 3))
    cov = a @ a.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def test_groups_from_sizes_partition() -> None:
    groups = maxvar.groups_from_sizes([3, 2, 4])
    assert [len(g) for g in groups] == [3, 2, 4]
    covered = np.concatenate(groups)
    assert sorted(covered) == list(range(9))


def test_maxvar_clean_identity_is_fixed_point() -> None:
    n = 9
    groups = maxvar.groups_from_sizes([3, 3, 3])
    cleaned = maxvar.maxvar_clean(np.eye(n), groups)
    np.testing.assert_allclose(cleaned, np.eye(n), atol=1e-8)


def test_maxvar_clean_output_is_valid_correlation_matrix() -> None:
    n = 12
    groups = maxvar.groups_from_sizes([5, 4, 3])
    corr = _random_correlation(n, seed=1)
    cleaned = maxvar.maxvar_clean(corr, groups)
    np.testing.assert_allclose(np.diag(cleaned), 1.0, atol=1e-8)
    np.testing.assert_allclose(cleaned, cleaned.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(cleaned)
    assert eigvals.min() > -1e-8


def test_maxvar_clean_rejects_non_partition() -> None:
    corr = _random_correlation(6, seed=2)
    bad_groups = [np.array([0, 1]), np.array([1, 2, 3])]  # overlap, incomplete
    with pytest.raises(AssertionError):
        maxvar.maxvar_clean(corr, bad_groups)


def test_maxvar_clean_reduces_distance_to_block_reference() -> None:
    """Sanity check of the paper's Sec. 3 claim: cleaning a noisy sample drawn
    from a reference with exact block (one-factor-per-group) structure moves
    the estimate closer to that reference than the raw empirical correlation
    matrix, for most draws."""
    n = 15
    groups = maxvar.groups_from_sizes([5, 5, 5])
    rng = np.random.default_rng(3)

    # A reference with the single-factor-per-group structure MaxVar assumes:
    # within-group correlation rho_in, zero correlation across groups.
    c0 = np.eye(n)
    rho_in = 0.6
    for g in groups:
        for i in g:
            for j in g:
                if i != j:
                    c0[i, j] = rho_in

    wins = 0
    trials = 20
    for trial in range(trials):
        samples = rng.multivariate_normal(np.zeros(n), c0, size=60)
        empirical = np.corrcoef(samples, rowvar=False)
        cleaned = maxvar.maxvar_clean(empirical, groups)
        dist_empirical = np.linalg.norm(empirical - c0, ord="fro")
        dist_cleaned = np.linalg.norm(cleaned - c0, ord="fro")
        wins += dist_cleaned < dist_empirical
    assert wins >= 0.8 * trials
