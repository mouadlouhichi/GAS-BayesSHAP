"""Exponential Hamming kernel and exact Shapley-covariance lemmas (D & E)."""

from .covariance import lemma_D_cross_cov, lemma_D_cross_cov_matrix, lemma_E_prior_cov
from .hamming import ExponentialHammingKernel, rho_from_lengthscale

__all__ = [
    "lemma_D_cross_cov",
    "lemma_D_cross_cov_matrix",
    "lemma_E_prior_cov",
    "ExponentialHammingKernel",
    "rho_from_lengthscale",
]
