"""Plain Monte-Carlo Shapley baseline (SamplingSHAP-style, stratum-uniform)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..game.oracle import CoalitionOracle
from ..game.subsets import random_subset


def monte_carlo_shapley(
    oracle: CoalitionOracle,
    x: np.ndarray,
    n_samples: int,
    rng: Optional[np.random.RandomState] = None,
) -> dict:
    """Stratified Monte-Carlo Shapley estimate (uniform over coalition sizes).

    For each sample: draw a uniform coalition size ``s``, a uniform coalition
    of that size, and compute add-one marginals for all features — the same
    per-player accounting as the residual sampler but without GP control
    variates or certification.  Query counts are reported.
    """
    rng = rng if rng is not None else np.random.RandomState(0)
    M = oracle.M
    sums = np.zeros(M, dtype=np.float64)
    start_coal = oracle.total_coalition_evals
    start_model = oracle.total_model_evals
    for _ in range(int(n_samples)):
        s = int(rng.randint(1, M))  # interior strata only (1..M-1)
        S = random_subset(rng, M, s)
        v_S = oracle.evaluate(x, S)
        for i in range(M):
            if not S[i]:
                S_u = S.copy()
                S_u[i] = True
                v_Su = oracle.evaluate(x, S_u)
                sums[i] += v_Su - v_S
    phi = sums / max(1, n_samples)
    return {
        "shapley_values": phi,
        "num_coalition_evals": oracle.total_coalition_evals - start_coal,
        "num_model_evals": oracle.total_model_evals - start_model,
        "n_samples": int(n_samples),
    }
