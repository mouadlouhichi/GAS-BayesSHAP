"""Shapley weights and coefficient utilities (project-structure shim).

The canonical implementation lives in :mod:`gas_bayesshap.game.subsets`
(weights, delta-weights, subset iteration) and
:mod:`gas_bayesshap.numerics.stable_combinatorics` (scalar weights).  This
module re-exports the same API under the ``game/shapley.py`` name required by
the project structure so imports from either location are equivalent.
"""

from .subsets import (
    delta_weights,
    shapley_weights,
)
from ..numerics.stable_combinatorics import (
    delta_weight,
    shapley_weight,
)

__all__ = [
    "shapley_weight",
    "shapley_weights",
    "delta_weight",
    "delta_weights",
]
