"""Tier-3: null-player certified containment."""

import numpy as np

from gas_bayesshap import GASBayesSHAP

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def test_null_player_certified_containment():
    M = 5
    weights = np.array([1.5, -2.0, 0.5, 3.0, 0.0])  # player 4 is null

    def model(x):
        return float(np.dot(x, weights) + 0.5 * x[0] * x[1])

    bg = np.zeros((5, M))
    eng = GASBayesSHAP(model, bg, output_bounds=(-5.0, 10.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=1.0, delta=0.05, max_budget=400)

    phi_null = res["shapley_values"][4]
    w_null = res["certified_projected_widths"][4]
    assert abs(phi_null - 0.0) <= w_null
    assert w_null > 0  # a real (non-degenerate) interval


def test_null_player_containment_multiple_runs():
    M = 4
    weights = np.array([1.0, 0.0, -1.0, 0.0])

    def model(x):
        return float(np.dot(x, weights))

    bg = np.zeros((3, M))
    for seed in range(5):
        eng = GASBayesSHAP(model, bg, output_bounds=(-2.0, 2.0),
                           rng=np.random.RandomState(seed), config=ENGINE_CONFIG)
        res = eng.explain(np.ones(M), epsilon=0.8, delta=0.05, max_budget=200)
        for null_idx in (1, 3):
            phi_n = res["shapley_values"][null_idx]
            w_n = res["certified_projected_widths"][null_idx]
            assert abs(phi_n) <= w_n
