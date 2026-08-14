"""GAS-BayesSHAP: Gaussian-Adaptive Stratified Bayesian Shapley Estimation (v11.0).

Bounded-linear Bayesian active control variates combined with
Neyman-stratified anytime empirical-Bernstein certification for
distribution-free Shapley estimation with certified confidence widths.

Implementation spec: ``specs/GAS_BayesSHAP_Implementation_Spec (4).md``.
"""

from ._version import __version__
from .core.estimator import GASBayesSHAP
from .core.results import ResultStatus, RunResults
from .game.oracle import CoalitionOracle
from .game.brute_force import (
    brute_force_shapley,
    brute_force_cross_covariance,
    brute_force_prior_covariance,
)

__all__ = [
    "GASBayesSHAP",
    "CoalitionOracle",
    "ResultStatus",
    "RunResults",
    "brute_force_shapley",
    "brute_force_cross_covariance",
    "brute_force_prior_covariance",
    "__version__",
]
