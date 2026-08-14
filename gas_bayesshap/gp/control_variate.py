"""Bounded linear control variate (Module A).

Spec section 2.1: to eliminate clipping bias while strictly guaranteeing
``m_b(S) in [L, U]`` for **all** :math:`2^M` coalitions:

.. math::

    h(S) = \\mathbf{k}_\\mathcal{D}(S)^T \\boldsymbol{\\alpha},
    \\qquad \\boldsymbol{\\alpha} = (\\mathbf{K}_{\\mathcal{D},\\mathcal{D}}
        + \\eta^2 \\mathbf{I})^{-1} \\mathbf{y}

Since :math:`k(S, S_j) \\in [\\sigma_0^2 \\rho^M, \\sigma_0^2]`:

.. math::

    h_{\\text{lb}} = \\sigma_0^2\\left(\\rho^M \\sum_{\\alpha_j>0}\\alpha_j
        + \\sum_{\\alpha_j<0}\\alpha_j\\right), \\qquad
    h_{\\text{ub}} = \\sigma_0^2\\left(\\sum_{\\alpha_j>0}\\alpha_j
        + \\rho^M \\sum_{\\alpha_j<0}\\alpha_j\\right)

    \\lambda = \\min\\left(1, \\frac{U - L}{h_{\\text{ub}} - h_{\\text{lb}}}\\right),
    \\qquad c = L - \\lambda h_{\\text{lb}}

The **bounded linear surrogate** is :math:`m_b(S) = c + \\lambda h(S)`.
The analytical surrogate Shapley attribution is

.. math::

    \\boldsymbol{\\phi}(m_b) = \\lambda \\mathbf{K}_{\\phi,\\mathcal{D}}
        \\boldsymbol{\\alpha}

and the posterior Shapley covariance is :math:`\\lambda^2 \\boldsymbol{\\Sigma}_h`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..kernels.covariance import (
    lemma_D_cross_cov_matrix,
    lemma_E_prior_cov,
)
from ..kernels.hamming import ExponentialHammingKernel
from ..numerics.validation import NumericalFailure, assert_finite
from .posterior import gp_posterior


@dataclass
class BoundedLinearSurrogate:
    """Frozen bounded-linear surrogate state (checkpointable)."""

    D_coalitions: np.ndarray            # (D, M) bool matrix
    D_y: np.ndarray                     # (D,) centred observations y = v - E_base
    inv_K_DD: np.ndarray                # (D, D)
    K_phi_D: np.ndarray                 # (M, D)
    alpha: np.ndarray                   # (D,)
    h_lb: float
    h_ub: float
    scale: float                        # lambda
    shift: float                        # c
    M: int
    eta: float

    def predict(self, S: np.ndarray, kernel: ExponentialHammingKernel, count: Optional[list] = None) -> float:
        from .posterior import gp_predict
        return gp_predict(self.D_coalitions, self.alpha, S, kernel, self.scale, self.shift, count_predictions=count)

    def surrogate_shapley(self, kernel: ExponentialHammingKernel) -> np.ndarray:
        """``phi(m_b) = lambda * K_phi_D @ alpha`` (no 2^M enumeration)."""
        return self.scale * (self.K_phi_D @ self.alpha)

    def posterior_covariance(self) -> np.ndarray:
        """``lambda^2 * (K_phi_phi - K_phi_D inv(K_DD) K_phi_D^T)``."""
        phi_cov, _ = gp_posterior(
            self._K_phi_phi,
            self.K_phi_D,
            self.inv_K_DD,
            scale=self.scale,
        )
        return phi_cov

    def posterior_variances(self, floor: float = 1e-10) -> np.ndarray:
        phi_cov = self.posterior_covariance()
        return np.maximum(np.diag(phi_cov), floor)

    # cached prior covariance is stored separately (see fit_bounded_surrogate)
    _K_phi_phi: np.ndarray = None  # type: ignore


def fit_bounded_surrogate(
    D_gp_coalitions: List[np.ndarray],
    D_gp_y: List[float],
    kernel: ExponentialHammingKernel,
    eta: float,
    K_phi_phi: Optional[np.ndarray] = None,
) -> BoundedLinearSurrogate:
    """Freeze the bounded linear surrogate from the active dataset.

    Parameters
    ----------
    D_gp_coalitions, D_gp_y:
        Active design (already centred, ``y = v - E_base``).
    kernel, eta:
        Kernel and jitter.
    K_phi_phi:
        Precomputed Lemma-E prior covariance; if None it is computed here.

    Returns
    -------
    BoundedLinearSurrogate with ``alpha``, ``D_matrix``, ``h_lb/h_ub``,
    ``scale`` (lambda) and ``shift`` (c).
    """
    if not D_gp_coalitions:
        raise NumericalFailure("fit_bounded_surrogate requires at least one observation")
    M = len(D_gp_coalitions[0])
    D = len(D_gp_coalitions)

    y_gp_vec = np.array(D_gp_y, dtype=np.float64)
    inv_K_DD = _inverse_from_list(D_gp_coalitions, kernel, eta)
    alpha = inv_K_DD @ y_gp_vec
    assert_finite(alpha, "alpha")

    D_matrix = np.array(D_gp_coalitions, dtype=bool)
    K_phi_D = lemma_D_cross_cov_matrix(kernel, D_matrix, M)
    if K_phi_phi is None:
        K_phi_phi = lemma_E_prior_cov(kernel, M)

    # Conservative global bounds on h(S)
    pos_alpha_sum = float(np.sum(alpha[alpha > 0]))
    neg_alpha_sum = float(np.sum(alpha[alpha < 0]))
    h_lb = kernel.sigma0_sq * ((kernel.rho ** M) * pos_alpha_sum + neg_alpha_sum)
    h_ub = kernel.sigma0_sq * (pos_alpha_sum + (kernel.rho ** M) * neg_alpha_sum)

    return BoundedLinearSurrogate(
        D_coalitions=D_matrix,
        D_y=y_gp_vec,
        inv_K_DD=inv_K_DD,
        K_phi_D=K_phi_D,
        alpha=alpha,
        h_lb=float(h_lb),
        h_ub=float(h_ub),
        scale=1.0,
        shift=0.0,
        M=M,
        eta=float(eta),
        _K_phi_phi=K_phi_phi,
    )


def apply_output_bounds(surrogate: BoundedLinearSurrogate, L: float, U: float) -> BoundedLinearSurrogate:
    """Apply the bounded-linear shrinkage: set lambda and c from (L, U)."""
    h_lb, h_ub = surrogate.h_lb, surrogate.h_ub
    if h_ub > h_lb:
        scale = min(1.0, (U - L) / (h_ub - h_lb))
    else:
        scale = 1.0
    shift = L - scale * h_lb
    surrogate.scale = float(scale)
    surrogate.shift = float(shift)
    return surrogate


def heuristic_output_bounds(E_base: float, v_N: float, delta_total: float):
    """Remark 2.2 fallback bounds when ``output_bounds=None``.

    ``L = min(E_base, v_N) - |delta_total|``,
    ``U = max(E_base, v_N) + |delta_total|``,
    ``R_delta_res = 4 * (U - L)`` (always >= the v7.1 heuristic).
    """
    L = min(E_base, v_N) - abs(delta_total)
    U = max(E_base, v_N) + abs(delta_total)
    return float(L), float(U)


def _inverse_from_list(coalitions: List[np.ndarray], kernel: ExponentialHammingKernel, eta: float) -> np.ndarray:
    """Direct Gram inverse (used only for the frozen fit; incremental path in the estimator)."""
    D = np.array(coalitions, dtype=bool)
    K = kernel.gram(D)
    inv = np.linalg.inv(K + eta ** 2 * np.eye(D.shape[0]))
    assert_finite(inv, "Gram inverse")
    return inv
