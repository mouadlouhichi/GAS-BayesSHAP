"""Parity: the production engine must reproduce the spec's inline v11.0
reference exactly (same seeds -> same values, widths and query counts)."""

import numpy as np
import pytest

from gas_bayesshap import GASBayesSHAP as ProductionGASBayesSHAP
from gas_bayesshap.reference.spec_v11_reference import GASBayesSHAP as ReferenceGASBayesSHAP

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def _run_pair(model, bg, bounds, x, seed, **kwargs):
    np.random.seed(seed)
    ref = ReferenceGASBayesSHAP(model, bg, output_bounds=bounds)
    r_ref = ref.explain(x, **kwargs)

    prod = ProductionGASBayesSHAP(model, bg, output_bounds=bounds,
                                  rng=np.random.RandomState(seed),
                                  config=ENGINE_CONFIG)
    r_prod = prod.explain(x, **kwargs)
    return r_ref, r_prod


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("M", [3, 4])
def test_parity_values_and_queries(seed, M):
    rng = np.random.RandomState(seed)
    w = rng.randn(M)
    bg = rng.randn(4, M)

    def model(x):
        return float(np.dot(x, w) + 0.3 * x[0] * x[1] if M > 1 else np.dot(x, w))

    x = np.ones(M)
    r_ref, r_prod = _run_pair(model, bg, (-5.0, 5.0), x, seed,
                              epsilon=0.5, delta=0.05, max_budget=200,
                              n_pilot=3, n_active_steps=10)

    for key in ("shapley_values", "surrogate_shapley", "residual_shapley",
                "raw_confidence_widths", "certified_projected_widths",
                "posterior_std"):
        assert np.allclose(r_ref[key], r_prod[key], atol=1e-9), f"{key} mismatch (seed={seed}, M={M})"

    assert r_ref["num_coalition_evals"] == r_prod["num_coalition_evals"], \
        (f"coalition eval mismatch: ref={r_ref['num_coalition_evals']} "
         f"prod={r_prod['num_coalition_evals']}")
    assert r_ref["num_model_evals"] == r_prod["num_model_evals"]
    assert r_ref["converged_early"] == r_prod["converged_early"]
    assert r_ref["certificate_is_rigorous"] == r_prod["certificate_is_rigorous"]
    assert r_ref["range_bound_is_heuristic"] == r_prod["range_bound_is_heuristic"]
    assert r_ref["uncertified_features"] == r_prod["uncertified_features"]


def test_parity_heuristic_bounds():
    rng = np.random.RandomState(5)
    M = 4
    w = rng.randn(M)

    def model(x):
        return float(np.dot(x, w))

    r_ref, r_prod = _run_pair(model, rng.randn(4, M), None, np.ones(M), 5,
                              epsilon=0.5, delta=0.05, max_budget=150,
                              n_pilot=2, n_active_steps=8)
    assert r_ref["range_bound_is_heuristic"] is True
    assert r_prod["range_bound_is_heuristic"] is True
    assert r_ref["certificate_is_rigorous"] is False
    assert r_prod["certificate_is_rigorous"] is False
    assert np.allclose(r_ref["shapley_values"], r_prod["shapley_values"], atol=1e-9)
    assert r_ref["num_coalition_evals"] == r_prod["num_coalition_evals"]


def test_parity_m2():
    r_ref, r_prod = _run_pair(
        lambda x: float(2 * x[0] + 3 * x[1]), np.zeros((2, 2)), (0.0, 5.0),
        np.ones(2), 0, epsilon=0.1, delta=0.05, max_budget=20, n_pilot=0, n_active_steps=0)
    assert np.allclose(r_ref["shapley_values"], r_prod["shapley_values"], atol=1e-12)
    assert np.array_equal(r_ref["raw_confidence_widths"], r_prod["raw_confidence_widths"])
    assert r_ref["num_coalition_evals"] == r_prod["num_coalition_evals"]
