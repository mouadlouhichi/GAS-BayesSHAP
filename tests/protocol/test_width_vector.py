"""Complete width-vector reporting (spec section 24)."""

import numpy as np

from gas_bayesshap import GASBayesSHAP

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def test_full_width_vector_returned():
    M = 5
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((3, M)),
                       output_bounds=(0.0, 5.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=0.5, delta=0.05, max_budget=150)
    W = res["raw_confidence_widths"]
    assert len(W) == M
    # every entry is present (finite or inf), never dropped
    assert np.all(np.isnan(W) == False)  # noqa: E712
    # max/mean/median/argmax available
    assert "max_width" in res
    assert "mean_width" in res
    assert "median_width" in res
    assert res["argmax_feature"] is not None
    assert res["max_width"] == float(np.max(W))


def test_converged_early_flag_matches_widths():
    eng = GASBayesSHAP(lambda x: float(2 * x[0] + 3 * x[1]), np.zeros((2, 2)),
                       output_bounds=(0.0, 5.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(2), epsilon=1.0, max_budget=20)
    assert res["converged_early"] == (np.max(res["raw_confidence_widths"]) <= 1.0)
    assert res["converged"] is True
