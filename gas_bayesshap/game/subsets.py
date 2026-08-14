"""Shapley weights and coalition subset utilities."""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np

from ..numerics.stable_combinatorics import delta_weight, shapley_weight


def shapley_weights(M: int) -> np.ndarray:
    """Vector of Shapley weights ``w_s = s!(M-1-s)! / M!`` for ``s = 0..M-1``."""
    return np.array([shapley_weight(s, M) for s in range(M)], dtype=np.float64)


def delta_weights(M: int) -> np.ndarray:
    """Vector of adjacent weight differences ``Delta w_s`` for ``s = 0..M-2``."""
    return np.array([delta_weight(s, M) for s in range(M - 1)], dtype=np.float64)


def all_subsets(M: int) -> Iterator[np.ndarray]:
    """Yield every coalition mask of ``{0..M-1}`` as a bool array (2^M)."""
    for mask in range(1 << M):
        yield np.array([(mask >> bit) & 1 for bit in range(M)], dtype=bool)


def all_bitmasks(M: int) -> Iterator[int]:
    for mask in range(1 << M):
        yield mask


def bitmask_to_mask(bitmask: int, M: int) -> np.ndarray:
    return np.array([(bitmask >> bit) & 1 for bit in range(M)], dtype=bool)


def mask_to_bitmask(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool)
    bitmask = 0
    for bit in range(len(m)):
        if m[bit]:
            bitmask |= 1 << bit
    return bitmask


def random_subset(rng: np.random.RandomState, M: int, s: int) -> np.ndarray:
    """Uniformly random coalition of size ``s`` (``rng.permutation``)."""
    p = np.zeros(M, dtype=bool)
    p[rng.permutation(M)[:s]] = True
    return p


def random_coalition(rng: np.random.RandomState, M: int) -> np.ndarray:
    """Random coalition with uniformly random size in ``[0, M]`` (acquisition pool)."""
    s_sz = rng.randint(0, M + 1)
    return random_subset(rng, M, s_sz)


def seed_coalitions(rng: np.random.RandomState, M: int) -> List[np.ndarray]:
    """Stage-1 seed design: ``{empty, full}`` plus one random subset per size.

    Exactly matches the spec reference: sizes ``1..M-1`` are drawn with
    ``rng.permutation``.
    """
    seeds = [np.zeros(M, dtype=bool), np.ones(M, dtype=bool)]
    for s in range(1, M):
        seeds.append(random_subset(rng, M, s))
    return seeds


def candidate_pool(rng: np.random.RandomState, M: int, pool_size: int) -> List[np.ndarray]:
    """Random acquisition candidate pool of ``pool_size`` coalitions.

    Uses the run RNG so the pool is deterministic for a given seed.
    """
    return [random_coalition(rng, M) for _ in range(int(pool_size))]


def default_pool_size(M: int) -> int:
    """``pool_size = max(32, 2*M)`` (configurable; not hard-coded inside engine)."""
    return max(32, 2 * M)


def default_refresh_interval(M: int) -> int:
    """Neyman refresh frequency: every ``5*M`` evaluations (spec Theorem A)."""
    return max(1, 5 * M)
