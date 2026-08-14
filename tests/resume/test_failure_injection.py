"""Failure injection: crash at various points -> safe resume or explicit
failure (spec section 51)."""

import numpy as np
import pytest

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.numerics.validation import NumericalFailure


BOUNDS = (-2.0, 2.0)
EPS = 9.0
BUDGET = 400


def _config(run_id, seed, run_dirs):
    return {
        "checkpoint_enabled": True,
        "checkpoint_every": 1,
        "cache_enabled": True,
        "persist_cache": True,
        "log_level": "NONE",
        "seed": seed,
        "results_dir": run_dirs["results_dir"],
        "checkpoints_dir": run_dirs["checkpoints_dir"],
        "run_id": run_id,
    }


def _game(M, seed):
    rng = np.random.RandomState(seed)
    w = rng.randn(M)
    bg = rng.randn(4, M)
    model = lambda x: float(np.dot(x, w) + 0.1 * np.sum(x ** 2))  # noqa: E731
    return model, bg


@pytest.mark.parametrize("crash_at_eval", [30, 60, 120])
def test_crash_during_residual_sampling_then_resume(crash_at_eval, run_dirs):
    M = 4
    seed = 11
    model, bg = _game(M, seed)

    # clean reference run
    clean = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                         model_tag="fi-model",
                         rng=np.random.RandomState(seed), config=_config("fi-clean", seed, run_dirs))
    res_clean = clean.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                              n_pilot=3, n_active_steps=10)
    assert res_clean["converged"] is True

    # crashing run: raise on the crash_at_eval-th coalition evaluation
    crashed = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                           model_tag="fi-model",
                           rng=np.random.RandomState(seed), config=_config("fi-crash", seed, run_dirs))
    state = {"n": 0}
    original_eval = crashed.oracle.evaluate

    def crashing_eval(x, S):
        state["n"] += 1
        if state["n"] == crash_at_eval:
            raise RuntimeError(f"injected crash at coalition eval {crash_at_eval}")
        return original_eval(x, S)

    crashed.oracle.evaluate = crashing_eval
    with pytest.raises(RuntimeError, match="injected crash"):
        crashed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                        n_pilot=3, n_active_steps=10)

    # resume with the healthy model
    resumed = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                           model_tag="fi-model",
                           rng=np.random.RandomState(seed), config=_config("fi-crash", seed, run_dirs))
    res_resumed = resumed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                                  n_pilot=3, n_active_steps=10, resume=True)
    assert res_resumed["converged"] is True
    assert np.allclose(res_resumed["shapley_values"], res_clean["shapley_values"], atol=1e-9)


def test_crash_before_any_checkpoint_restarts_cleanly(run_dirs):
    """Crash very early (no checkpoint yet): resume behaves like a fresh start."""
    M = 3
    seed = 2
    model, bg = _game(M, seed)

    def fragile(x):
        raise RuntimeError("crash immediately")

    with pytest.raises(RuntimeError):
        crashed = GASBayesSHAP(fragile, bg, output_bounds=BOUNDS,
                               rng=np.random.RandomState(seed), config=_config("fi-early", seed, run_dirs))
        crashed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=100)

    resumed = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                           rng=np.random.RandomState(seed), config=_config("fi-early", seed, run_dirs))
    res = resumed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=100, resume=True)
    assert res["converged"] is True  # fresh run completes


def test_nan_injection_fails_diagnostically(run_dirs):
    """NaN outputs must fail explicitly and diagnostically (spec section 51)."""
    M = 3
    calls = {"n": 0}

    def bad(x):
        calls["n"] += 1
        if calls["n"] > 50:
            return float("nan")
        return float(np.sum(x))

    eng = GASBayesSHAP(bad, np.zeros((2, M)), output_bounds=(0.0, 3.0),
                       rng=np.random.RandomState(0),
                       config={**_config("nan-run", 0, run_dirs), "cache_enabled": False,
                               "checkpoint_enabled": False})
    with pytest.raises(NumericalFailure, match="non-finite"):
        eng.explain(np.ones(M), epsilon=0.01, delta=0.05, max_budget=200)
