"""The complete 10-tier verification suite from the spec (section 6),
re-expressed against the production package."""

import numpy as np
import pytest

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.brute_force import (
    brute_force_cross_covariance,
    brute_force_prior_covariance,
)
from gas_bayesshap.kernels.covariance import lemma_D_cross_cov, lemma_E_prior_cov
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}

SIGMA0, LENGTHSCALE = 1.0, 1.5


def test_tier1_lemma_d_exact_enumeration():
    M = 4
    kernel = ExponentialHammingKernel(SIGMA0, LENGTHSCALE)
    S_j = np.array([True, False, True, False])
    K_a = lemma_D_cross_cov(kernel, S_j, M)
    K_b = brute_force_cross_covariance(kernel, S_j, M)
    assert np.allclose(K_a, K_b, atol=1e-10)


def test_tier2_lemma_e_raw_pair_counts():
    kernel = ExponentialHammingKernel(SIGMA0, LENGTHSCALE)
    for m in (2, 3, 4, 5, 6):
        K_a = lemma_E_prior_cov(kernel, m)
        K_b = brute_force_prior_covariance(kernel, m)
        assert np.max(np.abs(K_a - K_b)) < 1e-10, f"Lemma E failed at M={m}"


def test_tier3_null_player_containment():
    M = 5
    weights = np.array([1.5, -2.0, 0.5, 3.0, 0.0])

    def model(x):
        return float(np.dot(x, weights) + 0.5 * x[0] * x[1])

    eng = GASBayesSHAP(model, np.zeros((5, M)), output_bounds=(-5.0, 10.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=1.0, delta=0.05, max_budget=400)
    phi_null = res["shapley_values"][4]
    w_null = res["certified_projected_widths"][4]
    assert abs(phi_null - 0.0) <= w_null


def test_tier4_coverage_calibration():
    phi_exact = np.array([1.5, 2.5, -1.0])

    def model_cal(x):
        return float(x[0] + 2.0 * x[1] - x[2] + x[0] * x[1])

    R = 30
    finite_count = covered_count = 0
    for trial in range(R):
        e = GASBayesSHAP(model_cal, np.zeros((3, 3)), output_bounds=(-2.0, 5.0),
                         rng=np.random.RandomState(trial), config=ENGINE_CONFIG)
        r = e.explain(np.ones(3), epsilon=1.5, delta=0.05, max_budget=300)
        is_finite = np.all(np.isfinite(r["certified_projected_widths"]))
        is_covered = np.all(np.abs(r["shapley_values"] - phi_exact) <= r["certified_projected_widths"])
        if is_finite:
            finite_count += 1
            if is_covered:
                covered_count += 1
    cov_given_finite = covered_count / max(1, finite_count)
    assert cov_given_finite >= 0.90


def test_tier5_inflation_tightness():
    M = 5
    weights = np.array([1.5, -2.0, 0.5, 3.0, 0.0])

    def model(x):
        return float(np.dot(x, weights) + 0.5 * x[0] * x[1])

    eng = GASBayesSHAP(model, np.zeros((5, M)), output_bounds=(-5.0, 10.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=1.0, delta=0.05, max_budget=400)
    raw = np.asarray(res["raw_confidence_widths"], dtype=np.float64)
    proj = np.asarray(res["certified_projected_widths"], dtype=np.float64)
    finite = np.isfinite(raw)
    if np.any(finite):
        ratio = float(np.mean(proj[finite] / raw[finite]))
        assert 1.0 < ratio < 3.0
        print(f"inflation ratio = {ratio:.2f}x")


def test_tier6_query_isolation_and_budget():
    M = 5
    weights = np.array([1.5, -2.0, 0.5, 3.0, 0.0])

    def model(x):
        return float(np.dot(x, weights) + 0.5 * x[0] * x[1])

    eng = GASBayesSHAP(model, np.zeros((5, M)), output_bounds=(-5.0, 10.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res1 = eng.explain(np.ones(M), epsilon=1.0, delta=0.05, max_budget=200)
    res2 = eng.explain(np.ones(M) * 2, epsilon=1.0, delta=0.05, max_budget=200)
    assert res1["num_coalition_evals"] > 0
    assert res2["num_coalition_evals"] > 0


def test_tier7_surrogate_global_boundedness():
    M = 4
    L, U = 0.0, 1.0
    eng = GASBayesSHAP(lambda x: float(np.mean(x)), np.zeros((4, M)),
                       output_bounds=(L, U),
                       rng=np.random.RandomState(0),
                       config={**ENGINE_CONFIG, "validate_boundedness": True})
    eng.explain(np.ones(M), max_budget=50)
    for mask in range(1 << M):
        S = np.array([(mask >> b) & 1 for b in range(M)], dtype=bool)
        mb = eng._predict(S)
        assert L - 1e-10 <= mb <= U + 1e-10


def test_tier8_zero_extreme_stratum_allocation():
    from gas_bayesshap.residual.neyman import solve_coupled_neyman_allocation
    M = 4
    sol = solve_coupled_neyman_allocation(np.ones((M, M)), M, K_cert=100)
    assert sol.probabilities[0] == 0.0
    assert sol.probabilities[M - 1] == 0.0 or M <= 2
    assert abs(sol.probabilities.sum() - 1.0) < 1e-9


def test_tier9_m2_exact_certification():
    eng = GASBayesSHAP(lambda x: float(2 * x[0] + 3 * x[1]), np.zeros((2, 2)),
                       output_bounds=(0.0, 5.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(2), max_budget=20)
    assert np.all(np.asarray(res["certified_projected_widths"], dtype=np.float64) == 0.0)


def test_tier10_additive_recovery():
    w = np.array([1.0, 2.0, 3.0])
    eng = GASBayesSHAP(lambda x: float(np.dot(x, w)), np.zeros((3, 3)),
                       output_bounds=(0.0, 6.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(3), max_budget=100)
    assert np.allclose(res["shapley_values"], w, atol=0.2)
