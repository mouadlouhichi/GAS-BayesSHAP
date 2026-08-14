"""Deterministic candidate-pool generation (spec section 18).

``pool_size = max(32, 2*M)`` by default, generated with the run RNG so a
given seed yields a deterministic pool.  The size is configurable and never
hard-coded inside the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..game.subsets import random_coalition


def default_pool_size(M: int) -> int:
    return max(32, 2 * int(M))


@dataclass
class CandidatePool:
    M: int
    size: int
    rng: np.random.RandomState
    candidates: List[np.ndarray] = None  # type: ignore

    def __post_init__(self):
        self.size = int(self.size)
        self.candidates = [random_coalition(self.rng, self.M) for _ in range(self.size)]

    def iter_candidates(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)


def candidate_pool(rng: np.random.RandomState, M: int, pool_size: Optional[int] = None) -> List[np.ndarray]:
    """Return ``pool_size`` deterministic random coalitions (default max(32, 2M))."""
    size = default_pool_size(M) if pool_size is None else int(pool_size)
    return [random_coalition(rng, M) for _ in range(size)]
