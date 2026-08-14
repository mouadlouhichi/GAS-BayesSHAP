"""Near-duplicate handling in the active-GP acquisition loop."""

import numpy as np

from gas_bayesshap.gp.updates import rank1_inverse_update
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel


def test_active_loop_handles_repeated_draws():
    """Repeated coalition draws (guaranteed by the birthday bound over 2^M
    coalitions) must not corrupt the inverse: the loop completes, all matrices
    stay finite, and at least one duplicate draw occurs and is handled."""
    M = 4
    kernel = ExponentialHammingKernel(1.0, 1.5)
    eta_sq = (1e-4) ** 2
    rng = np.random.RandomState(3)

    coals = []
    inv = np.empty((0, 0), dtype=np.float64)
    seen = set()
    duplicate_draws = 0
    for _ in range(60):
        p = np.zeros(M, dtype=bool)
        p[rng.permutation(M)[: rng.randint(0, M + 1)]] = True
        key = tuple(p.tolist())
        if key in seen:
            duplicate_draws += 1
        else:
            seen.add(key)
        k_vec = np.array([kernel.k(p, S) for S in coals])
        new_inv, ok = rank1_inverse_update(inv, k_vec, kernel.k_self(), eta_sq)
        if ok:
            inv = new_inv
            coals.append(p)
    assert duplicate_draws > 0
    assert np.all(np.isfinite(inv))
    assert np.all(np.isfinite(inv @ np.ones(len(coals))))


def test_inverse_consistent_with_accepted_distinct_design():
    """After filtering out duplicate draws (as the seeds do), the incremental
    inverse equals the direct inverse to machine precision."""
    M = 4
    kernel = ExponentialHammingKernel(1.0, 1.5)
    eta_sq = (1e-4) ** 2
    rng = np.random.RandomState(5)
    seen = set()
    coals = []
    while len(coals) < 8:
        p = np.zeros(M, dtype=bool)
        p[rng.permutation(M)[: rng.randint(0, M + 1)]] = True
        key = tuple(p.tolist())
        if key not in seen:
            seen.add(key)
            coals.append(p)
    inv = np.empty((0, 0), dtype=np.float64)
    for i, S in enumerate(coals):
        k_vec = np.array([kernel.k(S, T) for T in coals[:i]])
        new_inv, ok = rank1_inverse_update(inv, k_vec, kernel.k_self(), eta_sq)
        assert ok
        inv = new_inv
    D = np.array(coals, dtype=bool)
    direct = np.linalg.inv(kernel.gram(D) + eta_sq * np.eye(len(coals)))
    assert np.allclose(inv, direct, atol=1e-8)
