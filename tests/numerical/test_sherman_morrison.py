"""Sherman-Morrison incremental inverse vs direct inversion."""

import numpy as np
import pytest

from gas_bayesshap.gp.updates import rank1_inverse_update, rank1_inverse_update_detailed
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel


def _distinct_design(M=5, D=8, seed=0):
    """Generate a design with pairwise-distinct coalitions."""
    rng = np.random.RandomState(seed)
    seen = set()
    coals = []
    while len(coals) < D:
        p = np.zeros(M, dtype=bool)
        p[rng.permutation(M)[: rng.randint(0, M + 1)]] = True
        key = tuple(p.tolist())
        if key not in seen:
            seen.add(key)
            coals.append(p)
    return coals


def test_incremental_equals_direct():
    kernel = ExponentialHammingKernel(1.0, 1.5)
    coals = _distinct_design()
    eta_sq = (1e-4) ** 2
    inv = np.empty((0, 0), dtype=np.float64)
    for i, S in enumerate(coals):
        k_vec = np.array([kernel.k(S, T) for T in coals[:i]])
        new_inv, ok = rank1_inverse_update(inv, k_vec, kernel.k_self(), eta_sq)
        assert ok
        inv = new_inv
        D = np.array(coals[: i + 1], dtype=bool)
        direct = np.linalg.inv(kernel.gram(D) + eta_sq * np.eye(i + 1))
        assert np.allclose(inv, direct, atol=1e-8), f"mismatch at step {i}"


def test_detailed_update_matches_plain():
    kernel = ExponentialHammingKernel(1.0, 1.5)
    coals = _distinct_design()
    inv = np.empty((0, 0), dtype=np.float64)
    for i, S in enumerate(coals):
        k_vec = np.array([kernel.k(S, T) for T in coals[:i]])
        upd = rank1_inverse_update_detailed(inv, k_vec, kernel.k_self(), (1e-4) ** 2)
        assert upd.ok and upd.action == "accepted"
        inv = upd.inv_K
    assert np.all(np.isfinite(inv))


def test_guard_rejects_subthreshold_schur():
    """The near-duplicate guard: any schur < eta^2 must be rejected with the
    inverse left unchanged (spec section 20)."""
    eta_sq = (1e-4) ** 2
    inv_K = np.array([[1e8]])  # makes schur strongly negative
    k_vec = np.array([1.0])
    upd = rank1_inverse_update_detailed(inv_K, k_vec, k_self=1.0, eta_sq=eta_sq)
    assert not upd.ok
    assert upd.action == "rejected_near_duplicate"
    assert upd.schur < upd.threshold
    assert np.array_equal(upd.inv_K, inv_K)

    # plain variant returns (unchanged, False)
    new_inv, ok = rank1_inverse_update(inv_K, k_vec, 1.0, eta_sq)
    assert not ok
    assert np.array_equal(new_inv, inv_K)


def test_exact_duplicate_semantics_match_reference():
    """Reference semantics: the 1st exact duplicate has schur ~ 2*eta^2
    (>= eta^2) and is therefore *accepted* by the spec's guard."""
    kernel = ExponentialHammingKernel(1.0, 1.5)
    S0 = np.array([True, False, False])
    eta_sq = (1e-4) ** 2
    inv, ok = rank1_inverse_update(np.empty((0, 0)), np.array([]), kernel.k_self(), eta_sq)
    assert ok
    k_vec = np.array([kernel.k(S0, S0)])
    upd = rank1_inverse_update_detailed(inv, k_vec, kernel.k_self(), eta_sq)
    assert upd.ok  # accepted by the reference guard
    assert upd.schur >= upd.threshold


def test_base_case_single_observation():
    kernel = ExponentialHammingKernel(1.0, 1.5)
    inv, ok = rank1_inverse_update(np.empty((0, 0)), np.array([]), kernel.k_self(), (1e-4) ** 2)
    assert ok
    expected = 1.0 / (kernel.k_self() + (1e-4) ** 2)
    assert abs(inv[0, 0] - expected) < 1e-15


def test_random_distinct_systems_multiple_seeds():
    for seed in range(5):
        kernel = ExponentialHammingKernel(1.0, 1.5)
        coals = _distinct_design(seed=seed)
        eta_sq = (1e-4) ** 2
        inv = np.empty((0, 0), dtype=np.float64)
        for i, S in enumerate(coals):
            k_vec = np.array([kernel.k(S, T) for T in coals[:i]])
            new_inv, ok = rank1_inverse_update(inv, k_vec, kernel.k_self(), eta_sq)
            assert ok
            inv = new_inv
        D = np.array(coals, dtype=bool)
        direct = np.linalg.inv(kernel.gram(D) + eta_sq * np.eye(len(coals)))
        assert np.allclose(inv, direct, atol=1e-8)
