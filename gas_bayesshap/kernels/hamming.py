"""Exponential Hamming kernel over coalition subsets.

.. math::

    k(S, T) = \\sigma_0^2 \\, \\rho^{|S \\triangle T|}, \\qquad
    \\rho = e^{-1/\\ell}
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


class ExponentialHammingKernel:
    """Exponential Hamming kernel on ``{0, 1}^M`` (spec section 9).

    Parameters
    ----------
    sigma0 : float
        Kernel amplitude :math:`\\sigma_0 > 0`.
    lengthscale : float
        Characteristic lengthscale :math:`\\ell > 0`; :math:`\\rho = e^{-1/\\ell}`.
    """

    def __init__(self, sigma0: float = 1.0, lengthscale: float = 1.5):
        if sigma0 <= 0:
            raise ValueError("sigma0 must be > 0")
        if lengthscale <= 0:
            raise ValueError("lengthscale must be > 0")
        self.sigma0 = float(sigma0)
        self.sigma0_sq = float(sigma0) ** 2
        self.lengthscale = float(lengthscale)
        self.rho = float(np.exp(-1.0 / lengthscale))

    # ------------------------------------------------------------------ #
    def hamming(self, S1: np.ndarray, S2: np.ndarray) -> int:
        return int(np.sum(np.asarray(S1, dtype=bool) != np.asarray(S2, dtype=bool)))

    def k(self, S1: np.ndarray, S2: np.ndarray) -> float:
        """Scalar kernel value :math:`k(S_1, S_2)`."""
        d_H = self.hamming(S1, S2)
        return self.sigma0_sq * (self.rho ** d_H)

    def k_self(self) -> float:
        """Diagonal value :math:`k(S, S) = \\sigma_0^2`."""
        return self.sigma0_sq

    def vectorized(self, D_matrix: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Kernel vector ``[k(S_obs, S)]`` for every observed row of ``D_matrix``.

        Vectorized (no Python loop over observations): O(DM) Hamming distances.
        """
        S = np.asarray(S, dtype=bool)
        if len(D_matrix) == 0:
            return np.zeros(0, dtype=np.float64)
        d_H = np.sum(np.asarray(D_matrix, dtype=bool) != S[None, :], axis=1)
        return self.sigma0_sq * (self.rho ** d_H)

    def gram(self, D_matrix: np.ndarray) -> np.ndarray:
        """Gram matrix ``K_DD`` for the rows of ``D_matrix`` (full, non-incremental)."""
        D = np.asarray(D_matrix, dtype=bool)
        n = D.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        d = D[:, None, :] != D[None, :, :]
        d_H = d.sum(axis=2)
        return self.sigma0_sq * (self.rho ** d_H)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ExponentialHammingKernel(sigma0={self.sigma0}, "
            f"lengthscale={self.lengthscale}, rho={self.rho:.6f})"
        )


def rho_from_lengthscale(lengthscale: float) -> float:
    return float(np.exp(-1.0 / float(lengthscale)))
