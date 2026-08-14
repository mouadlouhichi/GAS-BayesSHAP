"""Exact analytical Shapley-kernel covariance quantities.

Implements the two exact spectral-hypercube lemmas of the spec:

* **Lemma D** — the :math:`O(M^2)` cross-covariance
  :math:`[\\mathbf{K}_{\\phi, \\mathcal{D}}]_{i,j} = \\mathcal{A}_i[k(\\cdot, S_j)]`
  exploiting the ``V_in(r)`` / ``V_out(r)`` symmetry.
* **Lemma E** — the exact analytical prior Shapley covariance
  :math:`\\mathbf{K}_{\\phi,\\phi} = \\mathcal{A}_i\\mathcal{A}_j' k(S, T)`
  with the :math:`\\Delta w_s` weight-difference factorization and raw pair
  counting.

These functions reproduce, exactly, the arithmetic of the spec's certified
inline reference implementation.
"""

from __future__ import annotations

import numpy as np

from ..numerics.stable_combinatorics import comb, factorial
from .hamming import ExponentialHammingKernel


# --------------------------------------------------------------------------- #
# Lemma D
# --------------------------------------------------------------------------- #
def lemma_D_cross_cov(kernel: ExponentialHammingKernel, S_j: np.ndarray, M: int) -> np.ndarray:
    """Exact closed-form cross-covariance vector ``A_i[k(., S_j)]`` in O(M^2).

    For a coalition ``S_j`` of size ``r``, every ``i in S_j`` shares the value
    ``V_in(r)`` and every ``i notin S_j`` shares ``V_out(r)`` (spec Lemma D,
    implementation spec section 2.2).  ``V_in(0) == 0`` and ``V_out(M) == 0``.
    """
    S_j = np.asarray(S_j, dtype=bool)
    r = int(np.sum(S_j))
    K_phi_j = np.zeros(M, dtype=np.float64)

    def eval_cross_scalar(r_no_i: int, sign: float) -> float:
        total_sum = 0.0
        for s in range(M):
            denom = comb(M - 1, s)
            if denom == 0:
                continue
            l_min = max(0, s - (M - 1 - r_no_i))
            l_max = min(s, r_no_i)
            overlap_sum = 0.0
            for l in range(l_min, l_max + 1):
                num = comb(r_no_i, l) * comb(M - 1 - r_no_i, s - l)
                power = r_no_i + s - 2 * l
                overlap_sum += (num / denom) * (kernel.rho ** power)
            total_sum += overlap_sum
        return ((kernel.sigma0_sq) * (1.0 - kernel.rho) / M) * sign * total_sum

    V_in = eval_cross_scalar(r - 1, +1.0) if r > 0 else 0.0
    V_out = eval_cross_scalar(r, -1.0) if r < M else 0.0

    K_phi_j[S_j] = V_in
    K_phi_j[~S_j] = V_out
    return K_phi_j


def lemma_D_cross_cov_matrix(kernel: ExponentialHammingKernel, D_matrix: np.ndarray, M: int) -> np.ndarray:
    """Stack Lemma-D vectors for every observed coalition in ``D_matrix``.

    Returns ``(M, D)`` so that ``phi(m) = K_phi_D @ alpha``.
    """
    cols = [lemma_D_cross_cov(kernel, S, M).reshape(-1, 1) for S in np.asarray(D_matrix, dtype=bool)]
    if not cols:
        return np.empty((M, 0), dtype=np.float64)
    return np.hstack(cols)


# --------------------------------------------------------------------------- #
# Lemma E
# --------------------------------------------------------------------------- #
def lemma_E_prior_cov(kernel: ExponentialHammingKernel, M: int) -> np.ndarray:
    """Exact analytical prior Shapley covariance ``K_phi_phi`` (Lemma E).

    .. math::

        \\mathbf{K}_{\\phi,\\phi}
            = (V_{\\text{diag}} - V_{\\text{off}}) \\mathbf{I}_M
              + V_{\\text{off}} \\mathbf{1}_M \\mathbf{1}_M^T

    ``V_off`` uses the raw pair counts with the ``Delta w_s`` factors
    (``w(s) - w(s+1)``), exactly as in the spec reference implementation.
    """
    M = int(M)

    def w(s: int) -> float:
        return factorial(s) * factorial(M - 1 - s) / factorial(M)

    # 1. Diagonal variance V_diag
    v_diag_sum = 0.0
    for s in range(M):
        for t in range(M):
            denom_t = comb(M - 1, t)
            if denom_t == 0:
                continue
            l_min = max(0, s + t - (M - 1))
            l_max = min(s, t)
            for l in range(l_min, l_max + 1):
                num = comb(s, l) * comb(M - 1 - s, t - l)
                v_diag_sum += (num / denom_t) * (kernel.rho ** (s + t - 2 * l))
    v_diag = (2.0 * kernel.sigma0_sq * (1.0 - kernel.rho) / (M ** 2)) * v_diag_sum

    # 2. Off-diagonal covariance V_off (raw pair counts with Delta w factors)
    v_off = 0.0
    if M > 1:
        dw = [w(s) - w(s + 1) for s in range(M - 1)]
        for s in range(M - 1):
            for t in range(M - 1):
                l_min = max(0, s + t - (M - 2))
                l_max = min(s, t)
                for l in range(l_min, l_max + 1):
                    count = comb(M - 2, s) * comb(s, l) * comb(M - 2 - s, t - l)
                    v_off += dw[s] * dw[t] * count * (kernel.rho ** (s + t - 2 * l))
        v_off *= kernel.sigma0_sq * ((1.0 - kernel.rho) ** 2)

    mat = (v_diag - v_off) * np.eye(M) + v_off * np.ones((M, M))
    return 0.5 * (mat + mat.T)
