"""Empirical residual-range tightening (review task #2)."""

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.certification.empirical_range import (
    empirical_max_range,
    holdout_surrogate_error_bound,
    per_stratum_ranges,
)
from gas_bayesshap.residual.strata import StratumStore

ENGINE_CONFIG = {"checkpoint_enabled": False, "cache_enabled": False, "log_level": "NONE"}


def test_empirical_max_range_basic():
    r = empirical_max_range(np.array([0.1, -0.3, 0.2, -0.05]), safety_factor=2.0)
    assert abs(r - 0.6) < 1e-12  # 2 * 0.3
    assert empirical_max_range(np.array([])) > 0  # floor
    assert empirical_max_range(np.array([0.0, 0.0])) > 0  # floor when all zero


def test_per_stratum_ranges_spec_unchanged():
    store = StratumStore(4)
    R = per_stratum_ranges(store, 4, mode="spec", spec_range=4.0)
    assert np.all(R == 4.0)


def test_per_stratum_empirical_smaller():
    store = StratumStore(4)
    for i in range(4):
        store.append(i, 1, np.zeros(4, dtype=bool), "add_one", 0.02 * (i + 1), 0)
        store.append(i, 2, np.zeros(4, dtype=bool), "add_one", -0.03 * (i + 1), 0)
    R = per_stratum_ranges(store, 4, mode="empirical_max", safety_factor=2.0, spec_range=4.0)
    assert R[1] < 1.0 and R[2] < 1.0  # empirical << spec 4.0
    assert R[0] == 4.0 and R[3] == 4.0  # extremes keep spec (0 width anyway)


def test_holdout_bound_finite():
    v = np.array([0.5, 0.4, 0.6, 0.45, 0.55, 0.48])
    m = np.array([0.49, 0.41, 0.59, 0.46, 0.54, 0.47])
    b = holdout_surrogate_error_bound(v, m, delta=0.05)
    assert 0 < b < 1.0
    assert np.isfinite(b)


def test_engine_empirical_mode_flags_heuristic():
    M = 3
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, M)),
                       output_bounds=(0.0, 3.0), rng=np.random.RandomState(0),
                       config={**ENGINE_CONFIG, "range_mode": "empirical_max"})
    res = eng.explain(np.ones(M), epsilon=5.0, delta=0.05, max_budget=60, n_pilot=2)
    # empirical mode is always heuristic (approximation)
    assert res["range_bound_is_heuristic"] is True
    assert res["range_mode"] == "empirical_max"
    assert res["R_delta_res_effective"] < res["R_delta_res_spec"]


def test_engine_spec_mode_unchanged():
    M = 3
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, M)),
                       output_bounds=(0.0, 3.0), rng=np.random.RandomState(0),
                       config={**ENGINE_CONFIG, "range_mode": "spec"})
    res = eng.explain(np.ones(M), epsilon=5.0, delta=0.05, max_budget=60, n_pilot=2)
    assert res["range_mode"] == "spec"
    assert abs(res["R_delta_res_spec"] - 12.0) < 1e-9
    assert abs(res["R_delta_res_effective"] - 12.0) < 1e-9
    # rigorous bounds stay rigorous
    assert res["range_bound_is_heuristic"] is False
