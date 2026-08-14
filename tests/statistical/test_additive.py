"""Tier-10: additive ground-truth recovery."""

import numpy as np

from gas_bayesshap import GASBayesSHAP

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def test_additive_ground_truth_recovery():
    M = 3
    w = np.array([1.0, 2.0, 3.0])
    eng = GASBayesSHAP(lambda x: float(np.dot(x, w)), np.zeros((3, M)),
                       output_bounds=(0.0, 6.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), max_budget=100)
    assert np.allclose(res["shapley_values"], w, atol=0.2)


def test_additive_recovery_larger_M():
    M = 6
    rng = np.random.RandomState(7)
    w = rng.uniform(0.5, 2.0, M)
    bg = rng.randn(8, M) * 0.1
    eng = GASBayesSHAP(lambda x: float(np.dot(x, w)), bg,
                       output_bounds=(-3.0, 3.0 * M),  # hybrids can dip slightly below 0
                       rng=np.random.RandomState(7), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=0.15, delta=0.05, max_budget=600)
    assert np.allclose(res["shapley_values"], w, atol=0.3)
    # efficiency still holds exactly
    assert abs(np.sum(res["shapley_values"]) - res["delta_total"]) < 1e-9
