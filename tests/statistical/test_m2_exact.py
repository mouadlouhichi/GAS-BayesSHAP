"""Tier-9: M=2 exact certification (zero width)."""

import numpy as np

from gas_bayesshap import GASBayesSHAP

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def test_m2_exact_certification():
    eng = GASBayesSHAP(
        lambda x: float(2 * x[0] + 3 * x[1]),
        np.zeros((2, 2)),
        output_bounds=(0.0, 5.0),
        rng=np.random.RandomState(0),
        config=ENGINE_CONFIG,
    )
    res = eng.explain(np.ones(2), max_budget=20)
    widths = np.asarray(res["certified_projected_widths"], dtype=np.float64)
    assert np.all(widths == 0.0)
    assert res["converged_early"] is True
    assert res["certificate_is_rigorous"] is True
    # exact recovery
    assert np.allclose(np.asarray(res["shapley_values"]), [2.0, 3.0], atol=1e-6)


def test_m1_exact_certification():
    eng = GASBayesSHAP(
        lambda x: float(4.0 * x[0]),
        np.zeros((2, 1)),
        output_bounds=(0.0, 5.0),
        rng=np.random.RandomState(0),
        config=ENGINE_CONFIG,
    )
    res = eng.explain(np.ones(1), max_budget=10)
    assert np.all(np.isfinite(res["certified_projected_widths"]))
    assert np.allclose(res["shapley_values"], [4.0], atol=1e-9)
