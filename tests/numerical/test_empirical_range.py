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


def test_population_size_matches_binomial():
    from gas_bayesshap.certification.empirical_range import population_size
    import math
    for M in (3, 5, 11):
        for s in range(0, M):
            assert population_size(M, s) == math.comb(M - 1, s)
            assert population_size(M, s) >= 1


def test_coupon_delta1_monotone_in_counts():
    from gas_bayesshap.certification.empirical_range import (
        finite_population_coupon_delta1, population_size,
    )
    store = StratumStore(4)
    # one stratum s=1, population C(3,1)=3; n=10 -> (2/3)**10
    for i in range(4):
        for _ in range(10):
            store.append(i, 1, np.zeros(4, dtype=bool), "add_one", 0.1, 0)
    d1 = finite_population_coupon_delta1(store, 4)
    expected = 4 * (1.0 - 1.0 / 3.0) ** 10
    assert abs(d1 - expected) < 1e-9
    # more samples -> smaller delta1
    store2 = StratumStore(4)
    for i in range(4):
        for _ in range(20):
            store2.append(i, 1, np.zeros(4, dtype=bool), "add_one", 0.1, 0)
    assert finite_population_coupon_delta1(store2, 4) < d1


def test_coupon_threshold():
    from gas_bayesshap.certification.empirical_range import coupon_threshold
    # M=3, s=1: population C(2,1)=2; need (1/2)^n <= 0.001 -> n >= 10
    t = coupon_threshold(3, 1, 0.001)
    assert t == 10
    assert (0.5) ** t <= 0.001
    assert (0.5) ** (t - 1) > 0.001


def test_finite_population_mode_is_rigorous_and_tighter():
    M = 3
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, M)),
                       output_bounds=(0.0, 3.0), rng=np.random.RandomState(0),
                       config={**ENGINE_CONFIG, "range_mode": "finite_population"})
    res = eng.explain(np.ones(M), epsilon=5.0, delta=0.05, max_budget=60, n_pilot=2)
    # finite-population mode is NOT heuristic: the observed-support range is
    # rigorous at the realised level 1 - delta2 - delta1
    assert res["range_bound_is_heuristic"] is False
    assert res["range_mode"] == "finite_population"
    assert res["R_delta_res_effective"] < res["R_delta_res_spec"]
    # realised coupon budget and coverage level are reported
    d1 = res["finite_population_delta1"]
    lvl = res["finite_population_coverage_level"]
    assert 0.0 <= d1 < 1.0
    assert 0.0 < lvl <= 1.0
    assert abs(lvl - (1.0 - 0.025 - d1)) < 1e-9
    assert isinstance(res["finite_population_at_level_delta"], bool)


def test_finite_population_mode_coverage_m3():
    """Empirical coverage of the finite-population certificate on the M=3
    calibration game must not fall below the nominal level (sanity at R=150)."""
    import math
    def model_cal(x):
        return float(x[0] + 2.0 * x[1] - x[2] + x[0] * x[1])
    from gas_bayesshap.game.brute_force import brute_force_shapley
    from gas_bayesshap.benchmarking.metrics import coverage_report
    phi_true = brute_force_shapley(
        lambda S: model_cal(np.asarray(S, dtype=float)), 3)
    phis, widths = [], []
    for trial in range(150):
        eng = GASBayesSHAP(model_cal, np.zeros((3, 3)),
                           output_bounds=(-2.0, 5.0),
                           rng=np.random.RandomState(1000 + trial),
                           config={**ENGINE_CONFIG, "range_mode": "finite_population"})
        r = eng.explain(np.ones(3), epsilon=1.5, delta=0.05, max_budget=120, n_pilot=2)
        phis.append(r["shapley_values"])
        widths.append(r["certified_projected_widths"])
    rep = coverage_report(phis, widths, phi_true)
    # M=3 -> population size C(2,1)=2, coupon term 2^-n is tiny; nominal
    # level is 0.975; allow Monte-Carlo slack on the empirical check.
    assert rep["finite_width_rate"] >= 0.99
    assert rep["empirical_coverage"] >= 0.90
