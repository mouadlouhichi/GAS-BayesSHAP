"""Missing-stratum detection (spec sections 25, 44)."""

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.certification.bernstein import residual_widths
from gas_bayesshap.residual.strata import StratumStore

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def test_missing_interior_cell_yields_inf_width():
    M = 4
    store = StratumStore(M)
    # extremes populated, interior cells empty
    for i in range(M):
        store.append(i, 0, np.zeros(M, dtype=bool), "add_one", 0.1, 0)
        store.append(i, M - 1, np.ones(M, dtype=bool), "remove_one", 0.1, 0)
    sigma_res = np.zeros((M, M))
    widths = residual_widths(store, sigma_res, M, 0.05, 4.0)
    assert np.all(np.isinf(widths))
    # missing cells are explicit
    missing = store.missing_cells()
    assert len(missing) == M * (M - 2)


def test_strata_never_silently_ignored():
    """The store must track every cell explicitly (spec section 25)."""
    M = 3
    store = StratumStore(M)
    for s in range(M):
        for i in range(M):
            assert store.count(i, s) == 0
    store.append(0, 0, np.zeros(M, dtype=bool), "add_one", 1.0, 0)
    assert store.count(0, 0) == 1
    assert store.count(1, 0) == 0


def test_strict_mode_no_false_certification():
    """With insufficient interior coverage, the run must NOT report
    certification (never convert an uncertified result into a certified one)."""
    # n_pilot=0 + tiny budget => interior cells stay empty -> inf widths
    eng = GASBayesSHAP(lambda x: float(x[0] + x[1]), np.zeros((2, 3)),
                       output_bounds=(0.0, 2.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(3), epsilon=0.01, delta=0.05, max_budget=4, n_pilot=0)
    assert res["converged"] is False
    assert res["certificate_is_rigorous"] is False
    widths = np.asarray(res["raw_confidence_widths"], dtype=np.float64)
    assert np.all(np.isinf(widths))           # missing strata are explicit (inf)
    assert res["uncertified_features"] == list(range(3))
    # point estimates are flagged partial where cells are missing (audit
    # Medium 2): unobserved cells are never silently presented as complete
    assert not all(res["point_estimate_complete"])
    assert res["missing_cells_by_feature"]  # at least one feature has gaps
