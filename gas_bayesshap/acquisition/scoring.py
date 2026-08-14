"""Attribution-aware acquisition score (spec section 18-19).

For a candidate coalition ``S`` the score is the sum of squared posterior
attribution-covariance entries divided by the posterior coalition variance:

.. math::

    \\text{score}(S) =
    \\frac{\\sum_i [\\operatorname{Cov}(\\phi_i, v(S) \\mid \\mathcal{D}_{gp})]^2}
    {\\max(v_{post}(S), 10^{-8}) + \\eta^2}

with

.. math::

    v_{post}(S) = k(S,S) - \\mathbf{k}_D(S)^T \\mathbf{K}_{DD}^{-1}
        \\mathbf{k}_D(S), \\qquad
    \\operatorname{Cov}(\\boldsymbol{\\phi}, v(S))
        = \\mathbf{K}_{\\phi,\\{S\\}} - \\mathbf{K}_{\\phi,D}
        \\mathbf{K}_{DD}^{-1} \\mathbf{k}_D(S)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..kernels.covariance import lemma_D_cross_cov
from ..kernels.hamming import ExponentialHammingKernel


def acquisition_score(
    candidate: np.ndarray,
    D_gp_coalitions: List[np.ndarray],
    inv_K_DD: np.ndarray,
    K_phi_D: np.ndarray,
    kernel: ExponentialHammingKernel,
    eta: float,
) -> float:
    """A-optimal attribution-aware acquisition score for one candidate."""
    k_cand = np.array(
        [kernel.k(candidate, S_obs) for S_obs in D_gp_coalitions], dtype=np.float64
    )
    k_self = kernel.k_self()
    v_post_var = k_self - float(k_cand.T @ inv_K_DD @ k_cand)
    cov_phi = lemma_D_cross_cov(kernel, candidate, len(candidate)) - (
        K_phi_D @ inv_K_DD @ k_cand
    )
    score = float(np.sum(cov_phi ** 2)) / (max(v_post_var, 1e-8) + eta ** 2)
    return score


def best_candidate(
    candidates,
    D_gp_coalitions: List[np.ndarray],
    inv_K_DD: np.ndarray,
    K_phi_D: np.ndarray,
    kernel: ExponentialHammingKernel,
    eta: float,
):
    """Return ``(best_score, best_S)`` over the pool."""
    best_score = -1.0
    best_S = None
    for p in candidates:
        score = acquisition_score(p, D_gp_coalitions, inv_K_DD, K_phi_D, kernel, eta)
        if score > best_score:
            best_score = score
            best_S = p
    return best_score, best_S
