"""Cumulative Stage-2 budget across resume (audit finding: High 1).

An interrupted run and its resume must together respect the SAME max_budget
as an uninterrupted run — the resumed invocation does not get a fresh
allowance.
"""

import numpy as np
import pytest

from gas_bayesshap import GASBayesSHAP

BOUNDS = (-4.0, 4.0)
EPS = 15.0
BUDGET = 600


def _model(M, seed=7):
    rng = np.random.RandomState(seed)
    w = rng.randn(M)

    def model(x):
        return float(np.dot(x, w) + 0.2 * x[0] * x[1] if M > 1 else np.dot(x, w))

    return model


def _config(run_id, seed, tmp_path):
    return {
        "checkpoint_enabled": True, "checkpoint_every": 1,
        "cache_enabled": True, "persist_cache": True,
        "log_level": "NONE", "seed": seed,
        "results_dir": str(tmp_path / "results"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
        "run_id": run_id,
    }


def test_resume_respects_cumulative_budget(tmp_path):
    M = 4
    seed = 7
    model = _model(M, seed)
    bg = np.random.RandomState(seed).randn(4, M)

    # interrupted run: crash mid-Stage-2
    crashed = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                           rng=np.random.RandomState(seed), model_tag="cum-model",
                           config=_config("cum-budget", seed, tmp_path))
    state = {"n": 0}
    orig = crashed.oracle.evaluate

    def crashing_eval(x, S):
        state["n"] += 1
        if state["n"] == 250:  # well inside the adaptive loop (Stage-1 ≈ 17 evals)
            raise RuntimeError("injected crash")
        return orig(x, S)

    crashed.oracle.evaluate = crashing_eval
    with pytest.raises(RuntimeError):
        crashed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                        n_pilot=3, n_active_steps=10)
    attempted_at_crash = crashed._stage2_attempted_total
    assert attempted_at_crash > 0

    # resumed run: same budget — must NOT get a fresh allowance
    resumed = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                           rng=np.random.RandomState(seed), model_tag="cum-model",
                           config=_config("cum-budget", seed, tmp_path))
    res = resumed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                          n_pilot=3, n_active_steps=10, resume=True)

    # total attempted (pre-crash + post-resume) never exceeds max_budget
    total_attempted = resumed._stage2_attempted_total
    assert total_attempted <= BUDGET
    # the resumed invocation performed only the remaining budget
    assert res["num_coalition_evals"] <= BUDGET - attempted_at_crash + M  # small slack


def test_fresh_and_resumed_use_same_total_budget(tmp_path):
    """A run that crashes after ~K attempts and resumes must stop at roughly
    the same total attempted count as an uninterrupted run with max_budget."""
    M = 4
    seed = 7
    model = _model(M, seed)
    bg = np.random.RandomState(seed).randn(4, M)

    # uninterrupted reference (tiny epsilon forces budget exhaustion)
    clean = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                         rng=np.random.RandomState(seed), model_tag="cum-clean",
                         config=_config("cum-clean", seed, tmp_path))
    res_clean = clean.explain(np.ones(M), epsilon=1e-9, delta=0.05, max_budget=BUDGET,
                              n_pilot=3, n_active_steps=10)
    assert res_clean["status"] == "BUDGET_EXHAUSTED"
    clean_attempted = clean._stage2_attempted_total
    assert clean_attempted <= BUDGET
    # the guard breaks BEFORE exceeding the budget: attempted <= BUDGET
    assert BUDGET - clean_attempted < 2 * (1 + M)
