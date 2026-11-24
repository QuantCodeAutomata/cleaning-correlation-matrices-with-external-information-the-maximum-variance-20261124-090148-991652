"""Rotationally Invariant Estimator (RIE) of Bun, Bouchaud and Potters.

Custom implementation of the optimal nonlinear shrinkage estimator described
in Bun, Bouchaud, Potters, "Cleaning large correlation matrices: tools from
random matrix theory", Physics Reports 666 (2017), Sec. 5-6, as referenced by
the paper's methodology (Sec. 3) for the RIE baseline. No maintained PyPI
package implements this exact estimator ("pyriest" is not published on PyPI;
checked and confirmed absent), so it is coded directly from the Physics
Reports formulas: a discretized Hilbert transform of the empirical spectral
density is used to estimate the Stieltjes transform on the real axis, which
gives the optimal (oracle) eigenvalue shrinkage under the Marchenko-Pastur
null and an "epsilon" numerical-regularization scheme for eigenvalues outside
the Marchenko-Pastur support.
"""

from __future__ import annotations

import numpy as np

from src.config import RIE_BANDWIDTH


def _marchenko_pastur_edges(q: float) -> tuple[float, float]:
    lambda_minus = (1 - np.sqrt(q)) ** 2
    lambda_plus = (1 + np.sqrt(q)) ** 2
    return lambda_minus, lambda_plus


def rie_clean(
    corr: np.ndarray,
    n_obs: int,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Clean a correlation matrix with the Bun-Bouchaud-Potters RIE.

    Parameters
    ----------
    corr : (n, n) ndarray
        Empirical correlation matrix.
    n_obs : int
        Number of independent observations T used to estimate ``corr``.
    bandwidth : float, optional
        Lorentzian smoothing width eta used only for the imaginary part of
        the discretized Stieltjes transform (the kernel density estimate of
        the empirical spectral density, Physics Reports Sec. 5.2). Physics
        Reports Sec. 5.2 gives only the order ``eta = O(n**-0.5)`` for this
        width; the paper itself does not fix a bandwidth for the RIE
        baseline either. Defaults to ``RIE_BANDWIDTH`` (see src/config.py),
        a value of that order selected by a Monte Carlo sensitivity check on
        synthetic Marchenko-Pastur and block-structured inputs with a known
        reference matrix (``eta = n**-0.5`` itself under-resolves the
        isolated top eigenvalues produced by the group structure of
        Experiment 1 and leaves RIE's cleaning no better than the empirical
        matrix).

    Returns
    -------
    (n, n) ndarray
        Cleaned, unit-diagonal correlation matrix.

    Notes
    -----
    The oracle shrinkage xi_k = lambda_k / |1 - q + q*lambda_k*s(lambda_k)|^2
    (Physics Reports Eq. 5.12) needs s(lambda_k), the Stieltjes transform of
    the empirical spectral density evaluated just above the real axis:
    s(lambda_k) = lim_{eta->0+} (1/n) sum_j 1/(lambda_k + i*eta - lambda_j)
    (Eq. 5.8-5.10). Discretizing this at finite n and finite eta, the j=k
    term equals 1/(i*eta) = -i/eta: for a bandwidth wide enough to resolve
    the local spacing of an isolated top eigenvalue (a "spike" produced by a
    strong group factor, as in the block-structured reference matrix of
    this paper's Sec. 3), this spurious self-term dominates the sum and
    corrupts xi_k for exactly the eigenvalues the estimator most needs to
    shrink correctly, turning RIE's cleaning into a noise amplifier instead
    of a noise reducer. Dropping the j=k term from the sum (equivalent to
    letting eta_k -> 0 only in that one term, while keeping a finite
    smoothing width for j != k) removes this finite-n artifact while
    leaving the Lorentzian-kernel estimator of the remaining spectral
    density unchanged.
    """
    n = corr.shape[0]
    q = n / n_obs
    assert q > 0

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-10, None)

    eta = RIE_BANDWIDTH if bandwidth is None else bandwidth

    diff = eigvals.reshape(-1, 1) - eigvals.reshape(1, -1)  # lambda_i - lambda_j

    # Lorentzian-regularized Stieltjes transform 1/(diff + i*eta)
    # = (diff - i*eta) / (diff**2 + eta**2), summed over j != i (the j = i
    # term is zeroed out explicitly rather than left in with diff = 0).
    real_terms = diff / (diff ** 2 + eta ** 2)
    density_terms = eta / (diff ** 2 + eta ** 2) / np.pi
    np.fill_diagonal(real_terms, 0.0)
    np.fill_diagonal(density_terms, 0.0)
    real_part = real_terms.sum(axis=1) / (n - 1)
    density = density_terms.sum(axis=1) / (n - 1)
    stieltjes = real_part - 1j * np.pi * density

    xi = eigvals / np.abs(1 - q + q * eigvals * stieltjes) ** 2  # Eq. 5.12 (oracle xi_k)

    lambda_minus, _ = _marchenko_pastur_edges(q)
    if q > 1:
        # (q-1)/q eigenvalues are exactly zero in the population spectrum;
        # the corresponding shrunk eigenvalues collapse to the MP lower edge.
        xi = np.where(eigvals > lambda_minus / 10, xi, lambda_minus)

    xi = np.clip(xi, 1e-8, None)
    cleaned = (eigvecs * xi) @ eigvecs.T
    cleaned = (cleaned + cleaned.T) / 2.0

    diag = np.diag(cleaned)
    scale = np.sqrt(np.clip(diag, 1e-12, None))
    cleaned_corr = cleaned / np.outer(scale, scale)
    np.fill_diagonal(cleaned_corr, 1.0)
    return cleaned_corr
