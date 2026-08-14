"""Add-one / remove-one residual marginals (Lemma F, spec section 14).

.. math::

    \\text{ADD-ONE} (i \\notin S): &\\quad
        R_i(S) = [v(S \\cup \\{i\\}) - v(S)] - [m_b(S \\cup \\{i\\}) - m_b(S)], \\quad \\text{stratum} = |S| \\\\
    \\text{REMOVE-ONE} (i \\in S): &\\quad
        R_i(S) = [v(S) - v(S \\setminus \\{i\\})] - [m_b(S) - m_b(S \\setminus \\{i\\})], \\quad \\text{stratum} = |S| - 1

Both preserve conditional stratum uniformity (unbiasedness).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def add_one_residual(
    v_S: float,
    v_S_plus_i: float,
    m_S: float,
    m_S_plus_i: float,
) -> float:
    """Residual marginal for ``i notin S``."""
    return float((v_S_plus_i - v_S) - (m_S_plus_i - m_S))


def remove_one_residual(
    v_S: float,
    v_S_minus_i: float,
    m_S: float,
    m_S_minus_i: float,
) -> float:
    """Residual marginal for ``i in S``."""
    return float((v_S - v_S_minus_i) - (m_S - m_S_minus_i))


def sample_coalition(rng: np.random.RandomState, M: int, s_target: int) -> np.ndarray:
    """Uniformly random coalition of size ``s_target`` (stratum draw)."""
    from ..game.subsets import random_subset
    return random_subset(rng, M, s_target)
