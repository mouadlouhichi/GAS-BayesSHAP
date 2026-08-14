"""GP posterior prediction and covariance (spec sections 8 & 21)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..kernels.hamming import ExponentialHammingKernel
from ..numerics.linear_algebra import posterior_covariance as _posterior_cov
from ..numerics.validation import NumericalFailure, assert_finite


def gp_predict(
    D_matrix: np.ndarray,
    alpha: np.ndarray,
    S: np.ndarray,
    kernel: ExponentialHammingKernel,
    surrogate_scale: float = 1.0,
    surrogate_shift: float = 0.0,
    count_predictions: Optional[list] = None,
) -> float:
    """Fast vectorized O(DM) prediction of the bounded linear surrogate.

    .. math::

        m_b(S) = c + \\lambda\\, h(S), \\qquad
        h(S) = \\mathbf{k}_\\mathcal{D}(S)^T \\boldsymbol{\\alpha}

    No nonlinear clipping: boundedness is guaranteed structurally by the
    linear shrinkage (see :mod:`gas_bayesshap.gp.control_variate`).
    """
    if count_predictions is not None:
        count_predictions.append(1)
    if D_matrix is None or alpha is None or len(D_matrix) == 0:
        return 0.0
    d_H = np.sum(np.asarray(D_matrix, dtype=bool) != np.asarray(S, dtype=bool)[None, :], axis=1)
    k_vec = kernel.sigma0_sq * (kernel.rho ** d_H)
    h_val = float(k_vec @ np.asarray(alpha, dtype=np.float64))
    return float(surrogate_shift + surrogate_scale * h_val)


def gp_predict_vectorized(
    D_matrix: np.ndarray,
    alpha: np.ndarray,
    S_batch: np.ndarray,
    kernel: ExponentialHammingKernel,
    surrogate_scale: float = 1.0,
    surrogate_shift: float = 0.0,
) -> np.ndarray:
    """Batch prediction for a (K, M) array of coalitions."""
    S_batch = np.asarray(S_batch, dtype=bool)
    d_H = np.sum(np.asarray(D_matrix, dtype=bool)[None, :, :] != S_batch[:, None, :], axis=2)
    k = kernel.sigma0_sq * (kernel.rho ** d_H)  # (K, D)
    h = k @ np.asarray(alpha, dtype=np.float64)
    return surrogate_shift + surrogate_scale * h


def gp_posterior(
    K_phi_phi: np.ndarray,
    K_phi_D: np.ndarray,
    inv_K_DD: np.ndarray,
    scale: float = 1.0,
    variance_floor: float = 1e-10,
):
    """Posterior Shapley covariance ``lambda^2 (K_phi_phi - K_phi_D inv(K_DD) K_phi_D^T)``.

    Returns ``(phi_cov, posterior_variances)`` with the diagonal floored at
    ``variance_floor`` (spec reference behaviour).
    """
    phi_cov_h = _posterior_cov(K_phi_phi, K_phi_D, inv_K_DD)
    phi_cov_mb = (scale ** 2) * phi_cov_h
    posterior_variances = np.maximum(np.diag(phi_cov_mb), variance_floor)
    return phi_cov_mb, posterior_variances


def validate_surrogate_boundedness(
    D_matrix: np.ndarray,
    alpha: np.ndarray,
    kernel: ExponentialHammingKernel,
    scale: float,
    shift: float,
    M: int,
    L: float,
    U: float,
    tol: float = 1e-10,
) -> bool:
    """Check ``m_b(S) in [L, U]`` across all :math:`2^M` coalitions (validation)."""
    from ..game.subsets import all_subsets
    ok = True
    for S in all_subsets(M):
        val = gp_predict(D_matrix, alpha, S, kernel, scale, shift)
        if val < L - tol or val > U + tol:
            ok = False
            break
    return ok
