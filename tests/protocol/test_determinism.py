"""Deterministic repeated run with identical state (spec sections 38, 44)."""

import numpy as np

from gas_bayesshap import GASBayesSHAP

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def _make_engine(seed):
    def model(x):
        return float(np.dot(x, np.array([1.0, -2.0, 0.5, 1.5])) + 0.2 * x[0] * x[3])

    return GASBayesSHAP(model, np.zeros((4, 4)), output_bounds=(-5.0, 5.0),
                        rng=np.random.RandomState(seed), config=ENGINE_CONFIG)


def test_deterministic_repeated_run():
    r1 = _make_engine(42).explain(np.ones(4), epsilon=0.5, delta=0.05, max_budget=200)
    r2 = _make_engine(42).explain(np.ones(4), epsilon=0.5, delta=0.05, max_budget=200)
    for key in ("shapley_values", "surrogate_shapley", "residual_shapley",
                "raw_confidence_widths", "certified_projected_widths"):
        assert np.allclose(r1[key], r2[key], atol=1e-12), key
    assert r1["num_coalition_evals"] == r2["num_coalition_evals"]
    assert r1["num_model_evals"] == r2["num_model_evals"]


def test_different_seeds_differ():
    r1 = _make_engine(1).explain(np.ones(4), epsilon=0.5, delta=0.05, max_budget=200)
    r2 = _make_engine(2).explain(np.ones(4), epsilon=0.5, delta=0.05, max_budget=200)
    assert not np.allclose(r1["shapley_values"], r2["shapley_values"])
