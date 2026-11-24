"""Maximum Variance (MaxVar) correlation-cleaning estimator.

Custom implementation of paper Sec. 3 ("The Maximum Variance estimator").
No library implements this group-constrained eigenvector estimator, so it is
coded directly from the paper's steps (a)-(f).

Given K disjoint (or overlapping) groups of assets, the estimator:
  (a) finds, for each group, the maximum-variance portfolio (leading
      eigenvector of the correlation matrix restricted to that group,
      embedded in R^n with zeros outside the group);
  (b) stacks these K portfolios into P = (p_1, ..., p_K), and forms
      G = P^T P, H = P^T C P;
  (c) solves the generalized eigenvalue problem on G^{-1/2} H G^{-1/2};
  (d) builds the constrained eigenvectors Z = P G^{-1/2} Omega;
  (e) completes the matrix with an isotropic component on the orthogonal
      complement of span(Z), with amplitude delta chosen so that
      Tr(F) = n;
  (f) renormalises F to a correlation matrix.
"""

from __future__ import annotations

import numpy as np


def _group_leading_eigenvector(corr: np.ndarray, group_idx: np.ndarray) -> np.ndarray:
    """Maximum-variance portfolio of one group, embedded in R^n (paper step a)."""
    n = corr.shape[0]
    sub = corr[np.ix_(group_idx, group_idx)]
    eigvals, eigvecs = np.linalg.eigh(sub)
    leading = eigvecs[:, np.argmax(eigvals)]
    portfolio = np.zeros(n)
    portfolio[group_idx] = leading
    norm = np.linalg.norm(portfolio)
    if norm > 0:
        portfolio /= norm
    return portfolio


def maxvar_clean(corr: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Clean a correlation matrix with the MaxVar estimator.

    Parameters
    ----------
    corr : (n, n) ndarray
        Empirical (or otherwise noisy) correlation matrix.
    groups : list of 1-D int ndarrays
        Index arrays partitioning ``range(n)`` into K groups (paper's three
        asset classes: stock indices, bonds, currencies).

    Returns
    -------
    (n, n) ndarray
        Cleaned, unit-diagonal correlation matrix F_hat (paper Eq. in step f).
    """
    n = corr.shape[0]
    k = len(groups)
    assert k >= 1, "at least one group is required"
    covered = np.concatenate(groups)
    assert len(covered) == len(np.unique(covered)) == n, "groups must partition all n assets"

    # Step (a): one maximum-variance portfolio per group.
    portfolios = [_group_leading_eigenvector(corr, g) for g in groups]
    P = np.column_stack(portfolios)  # (n, K)

    # Step (b)
    G = P.T @ P
    H = P.T @ corr @ P

    # Step (c): generalized eigenvalue problem H a = mu G a, solved by
    # symmetrising via G^{-1/2}.
    g_eigvals, g_eigvecs = np.linalg.eigh(G)
    g_eigvals = np.clip(g_eigvals, 1e-12, None)
    G_inv_sqrt = g_eigvecs @ np.diag(1.0 / np.sqrt(g_eigvals)) @ g_eigvecs.T
    symmetrized = G_inv_sqrt @ H @ G_inv_sqrt
    mu, omega = np.linalg.eigh(symmetrized)  # M = diag(mu), Omega orthogonal

    # Step (d): constrained eigenvectors.
    Z = P @ G_inv_sqrt @ omega  # (n, K)

    # Step (e): isotropic completion on the orthogonal complement of span(Z).
    ZZt = Z @ Z.T
    delta = (n - mu.sum()) / (n - k)
    F = Z @ np.diag(mu) @ Z.T + delta * (np.eye(n) - ZZt)
    F = (F + F.T) / 2.0  # enforce exact symmetry against round-off

    # Step (f): renormalise to a correlation matrix.
    diag = np.diag(F)
    assert np.all(diag > 0), "MaxVar completion produced a non-positive diagonal"
    scale = np.sqrt(diag)
    F_hat = F / np.outer(scale, scale)
    np.fill_diagonal(F_hat, 1.0)
    return F_hat


def groups_from_sizes(sizes: list[int]) -> list[np.ndarray]:
    """Build disjoint, contiguous index groups from a list of group sizes."""
    groups = []
    start = 0
    for size in sizes:
        groups.append(np.arange(start, start + size))
        start += size
    return groups
