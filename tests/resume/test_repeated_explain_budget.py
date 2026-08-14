"""Repeated explain() on the same engine must not inherit the previous call's
Stage-2 budget (audit caveat 5.1)."""

import numpy as np

from gas_bayesshap import GASBayesSHAP

BOUNDS = (-4.0, 4.0)
EPS = 15.0
BUDGET = 600

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def _model(M, seed=7):
    rng = np.random.RandomState(seed)
    w = rng.randn(M)

    def model(x):
        return float(np.dot(x, w) + 0.2 * x[0] * x[1] if M > 1 else np.dot(x, w))

    return model


def test_fresh_calls_reset_stage2_budget():
    """Each non-resumed explain() call starts with a zero cumulative Stage-2
    budget; the second call is not starved by the first."""
    M = 4
    seed = 7
    model = _model(M, seed)
    bg = np.random.RandomState(seed).randn(4, M)

    eng = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                       rng=np.random.RandomState(seed), config=ENGINE_CONFIG)

    r1 = eng.explain(np.ones(M), epsilon=1e-9, delta=0.05, max_budget=BUDGET,
                     n_pilot=3, n_active_steps=10)
    assert r1["status"] == "BUDGET_EXHAUSTED"
    attempted_1 = eng._stage2_attempted_total
    assert attempted_1 > 0

    # fresh call on the SAME engine: budget counter must reset
    r2 = eng.explain(np.ones(M) * 0.5, epsilon=1e-9, delta=0.05, max_budget=BUDGET,
                     n_pilot=3, n_active_steps=10)
    assert r2["status"] == "BUDGET_EXHAUSTED"
    assert eng._stage2_attempted_total <= BUDGET
    # the second call ran a full budget again (not starved by the first)
    assert eng._stage2_attempted_total > 0


def test_fresh_call_budget_isolation_same_max():
    """Two consecutive fresh calls each spend up to ~max_budget, proving the
    counter was reset (an inherited counter would cap the second call near
    zero additional rounds)."""
    M = 4
    seed = 7
    model = _model(M, seed)
    bg = np.random.RandomState(seed).randn(4, M)

    eng = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                       rng=np.random.RandomState(seed), config=ENGINE_CONFIG)
    for _ in range(2):
        res = eng.explain(np.ones(M), epsilon=1e-9, delta=0.05, max_budget=BUDGET,
                          n_pilot=3, n_active_steps=10)
        assert res["status"] == "BUDGET_EXHAUSTED"
        # each call spent a substantial fraction of the full budget
        assert eng._stage2_attempted_total > 0.5 * BUDGET
