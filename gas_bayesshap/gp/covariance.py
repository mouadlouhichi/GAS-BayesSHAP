"""GP covariance helpers (project-structure shim).

The canonical implementation lives in
:mod:`gas_bayesshap.kernels.covariance` (Lemma D cross-covariance and
Lemma E prior Shapley covariance) and
:mod:`gas_bayesshap.numerics.linear_algebra` (posterior covariance).
This module re-exports the same API under the ``gp/covariance.py`` name
required by the project structure.
"""

from ..kernels.covariance import (
    lemma_D_cross_cov,
    lemma_D_cross_cov_matrix,
    lemma_E_prior_cov,
)
from ..numerics.linear_algebra import posterior_covariance

__all__ = [
    "lemma_D_cross_cov",
    "lemma_D_cross_cov_matrix",
    "lemma_E_prior_cov",
    "posterior_covariance",
]
