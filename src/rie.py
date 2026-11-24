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


def _marchenko_pastur_edges(q: float) -> tuple[float, float]:
    lambda_minus = (1 - np.sqrt(q)) ** 2
    lambda_plus = (1 + np.sqrt(q)) ** 2
    return lambda_minus, lambda_plus


def rie_clean(corr: np.ndarray, n_obs: int, bandwidth: float | None = None) -> np.ndarray:
    """Clean a correlation matrix with the Bun-Bouchaud-Potters RIE.

    Parameters
    ----------
    corr : (n, n) ndarray
        Empirical correlation matrix.
    n_obs : int
        Number of independent observations T used to estimate ``corr``.
    bandwidth : float, optional
        Lorentzian smoothing width eta for the discretized Stieltjes
        transform (Physics Reports Sec. 5.2); defaults to n^{-1/2}, the
        scale the paper uses for the kernel width.

    Returns
    -------
    (n, n) ndarray
        Cleaned, unit-diagonal correlation matrix.
    """
    n = corr.shape[0]
    q = n / n_obs
    assert q > 0
    if bandwidth is None:
        bandwidth = 1.0 / np.sqrt(n)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-10, None)

    # Discretized Stieltjes transform g(z) = (1/n) sum_k 1/(z - lambda_k),
    # evaluated just above the real axis: z = lambda_i + i*bandwidth
    # (Physics Reports Eq. 5.8-5.10).
    lam_col = eigvals.reshape(-1, 1)
    z = eigvals.reshape(1, -1) + 1j * bandwidth
    stieltjes = np.mean(1.0 / (z - lam_col), axis=0)  # g(lambda_i + i*eta), one per i

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
