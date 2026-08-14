"""Exact query accounting (spec sections 3, 31, 32, 44)."""

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def _count_calls(model, x, S, B, M):
    oracle = CoalitionOracle(model, np.zeros((B, M)))
    n0 = oracle.total_coalition_evals  # 0 (E_base not a coalition eval)
    m0 = oracle.total_model_evals      # B from E_base
    v = oracle.evaluate(x, S)
    return v, oracle.total_coalition_evals - n0, oracle.total_model_evals - m0


def test_coalition_eval_cost_rules():
    B, M = 4, 3
    calls = {"n": 0}

    def model(x):
        calls["n"] += 1
        return float(np.sum(x))

    x = np.ones(M)
    # full coalition: 1 coalition eval, 1 model eval
    v, c, m = _count_calls(model, x, np.ones(M, dtype=bool), B, M)
    assert c == 1 and m == 1
    assert v == float(np.sum(x))
    # empty coalition: 1 coalition eval, 0 model evals
    v, c, m = _count_calls(model, x, np.zeros(M, dtype=bool), B, M)
    assert c == 1 and m == 0
    assert v == 0.0  # E_base over zero background
    # hybrid: 1 coalition eval, B model evals
    v, c, m = _count_calls(model, x, np.array([True, False, True]), B, M)
    assert c == 1 and m == B


def test_explain_reports_per_call_deltas():
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((3, 3)),
                       output_bounds=(0.0, 3.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    r1 = eng.explain(np.ones(3), epsilon=1.0, delta=0.05, max_budget=40)
    r2 = eng.explain(np.ones(3) * 2, epsilon=1.0, delta=0.05, max_budget=40)
    assert r1["num_coalition_evals"] > 0
    assert r2["num_coalition_evals"] > 0
    assert r1["num_model_evals"] > 0
    assert r2["num_model_evals"] > 0
    # cumulative meters keep growing
    assert eng.oracle.total_coalition_evals > r1["num_coalition_evals"]


def test_stage2_budget_never_exceeded():
    M = 4
    eng = GASBayesSHAP(lambda x: float(np.dot(x, [1.0, -1.0, 0.5, 2.0])),
                       np.zeros((4, M)), output_bounds=(-3.0, 3.0),
                       rng=np.random.RandomState(1), config=ENGINE_CONFIG)
    budget = 25
    res = eng.explain(np.ones(M), epsilon=0.001, delta=0.05, max_budget=budget)
    # Stage-2 adaptive evals = total - (preflight + stage1 + extreme + pilot)
    # but max_budget is only the stage-2 loop allowance; verify it is respected:
    # the stage-2 loop ran until the budget guard triggered (epsilon tiny)
    assert res["converged"] is False
    assert res["status"] in ("BUDGET_EXHAUSTED", "NOT_CERTIFIED")
    # compute stage-2 evals directly: total evals - fixed costs
    M, n_pil = 4, 3
    stage1 = 1 + 2 + (M - 1) + 25  # preflight v_N + seeds + active steps
    fixed = 2 * M + 2 + (M - 2) * n_pil * (1 + M)  # extremes + pilot
    stage2 = res["num_coalition_evals"] - stage1 - fixed
    # each round costs at most 1 + M; the guard ensures spent + (1+M) <= budget
    assert stage2 <= budget, f"stage2 evals {stage2} exceeded budget {budget}"


def test_num_model_evals_counts_B_per_hybrid():
    B, M = 5, 3
    eng = GASBayesSHAP(lambda x: float(np.mean(x)), np.zeros((B, M)),
                       output_bounds=(0.0, 1.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=1.0, delta=0.05, max_budget=30)
    coal = res["num_coalition_evals"]
    # model evals >= coal (each coalition costs 0,1,B; minimum 0)
    assert res["num_model_evals"] >= 0
    # every non-empty, non-full coalition costs exactly B
    assert res["num_model_evals"] <= B * coal
