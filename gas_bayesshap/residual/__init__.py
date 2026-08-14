"""Stratified residual estimation (Module B)."""

from .estimator import residual_shapley
from .neyman import NeymanSolution, solve_coupled_neyman_allocation
from .sampling import add_one_residual, remove_one_residual
from .strata import StratumStore, ResidualRecord

__all__ = [
    "residual_shapley",
    "NeymanSolution",
    "solve_coupled_neyman_allocation",
    "add_one_residual",
    "remove_one_residual",
    "StratumStore",
    "ResidualRecord",
]
