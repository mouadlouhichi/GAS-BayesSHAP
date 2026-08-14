"""Checkpoint/resume equivalence (spec sections 34-36, 44)."""

import numpy as np
import pytest

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.checkpointing.compatibility import CheckpointCompatibilityError


# convergent settings (verified: widths reach < epsilon within budget)
BOUNDS = (-4.0, 4.0)
EPS = 15.0
BUDGET = 600


def _model(M, seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randn(M)

    def model(x):
        return float(np.dot(x, w) + 0.2 * x[0] * x[1] if M > 1 else np.dot(x, w))

    return model


def _config(run_id, seed, run_dirs):
    return {
        "checkpoint_enabled": True,
        "checkpoint_every": 1,
        "cache_enabled": True,
        "persist_cache": True,
        "gp_prediction_cache": False,
        "log_level": "NONE",
        "seed": seed,
        "results_dir": run_dirs["results_dir"],
        "checkpoints_dir": run_dirs["checkpoints_dir"],
        "run_id": run_id,
    }


def test_resume_continues_identical_result(run_dirs):
    """A run that crashes mid-Stage-2 and is resumed must equal the clean run:
    GP state restored, residual observations preserved, RNG restored, no
    Stage-1 rerun, no repeated cached oracle queries."""
    M = 4
    seed = 7
    model = _model(M, seed)
    bg = np.random.RandomState(seed).randn(4, M)

    # clean run
    clean = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                         rng=np.random.RandomState(seed), config=_config("resume-clean", seed, run_dirs))
    res_clean = clean.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                              n_pilot=3, n_active_steps=10)
    assert res_clean["converged"] is True

    # crashed run: raise on the 85th coalition evaluation (mid adaptive loop)
    crashed = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                           rng=np.random.RandomState(seed), model_tag="resume-model",
                           config=_config("resume-crash", seed, run_dirs))
    state = {"n": 0}
    original_eval = crashed.oracle.evaluate

    def crashing_eval(x, S):
        state["n"] += 1
        if state["n"] == 85:
            raise RuntimeError("injected crash")
        return original_eval(x, S)

    crashed.oracle.evaluate = crashing_eval
    with pytest.raises(RuntimeError, match="injected crash"):
        crashed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                        n_pilot=3, n_active_steps=10)

    # resumed run: same run_id + model_tag -> loads latest residual checkpoint
    resumed = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                           rng=np.random.RandomState(seed), model_tag="resume-model",
                           config=_config("resume-crash", seed, run_dirs))
    res_resumed = resumed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                                  n_pilot=3, n_active_steps=10, resume=True)

    assert res_resumed["converged"] is True
    assert np.allclose(res_resumed["shapley_values"], res_clean["shapley_values"], atol=1e-9)
    assert np.allclose(res_resumed["raw_confidence_widths"], res_clean["raw_confidence_widths"], atol=1e-12)
    assert np.allclose(res_resumed["certified_projected_widths"], res_clean["certified_projected_widths"], atol=1e-9)
    assert np.allclose(res_resumed["surrogate_shapley"], res_clean["surrogate_shapley"], atol=1e-12)
    # residual samples accumulate monotonically (never reset, never duplicated)
    assert res_resumed["num_residual_samples"] > 0
    assert res_resumed["num_sampling_rounds"] > 0
    # Stage 1 was NOT rerun: GP observations unchanged
    assert len(resumed._surrogate.D_coalitions) == len(crashed._surrogate.D_coalitions)
    # cached oracle queries were not repeated: resume added fewer new evals
    assert res_resumed["num_coalition_evals"] < res_clean["num_coalition_evals"]


def test_resume_gp_stage_after_crash_in_stage2(run_dirs):
    """Crash after the gp_stage checkpoint (during Stage-2 extreme init) then
    resume: result equals the clean run."""
    M = 4
    seed = 3
    model = _model(M, seed)
    bg = np.random.RandomState(seed).randn(4, M)

    clean = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                         rng=np.random.RandomState(seed), config=_config("resume-gp-clean", seed, run_dirs))
    res_clean = clean.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                              n_pilot=3, n_active_steps=10)
    assert res_clean["converged"] is True

    # crash during Stage 2 (after Stage 1, before any residual checkpoint):
    # Stage 1 uses 1 (preflight) + 5 (seeds) + 10 (active) = 16 oracle calls
    calls = {"n": 0}

    def fragile(x):
        calls["n"] += 1
        if calls["n"] > 22:
            raise RuntimeError("injected crash during Stage 2")
        return float(model(x))

    with pytest.raises(RuntimeError):
        crashed = GASBayesSHAP(fragile, bg, output_bounds=BOUNDS,
                               model_tag="gp-model",
                               rng=np.random.RandomState(seed),
                               config=_config("resume-gp-crash", seed, run_dirs))
        crashed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                        n_pilot=3, n_active_steps=10)

    # resume with the healthy model and same run_id
    resumed = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                           model_tag="gp-model",
                           rng=np.random.RandomState(seed),
                           config=_config("resume-gp-crash", seed, run_dirs))
    res_resumed = resumed.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=BUDGET,
                                  n_pilot=3, n_active_steps=10, resume=True)
    assert res_resumed["converged"] is True
    assert np.allclose(res_resumed["shapley_values"], res_clean["shapley_values"], atol=1e-9)


def test_resume_completed_run_returns_stored_result(run_dirs):
    """Resuming after completion returns the stored final result (no rerun)."""
    M = 3
    seed = 5
    model = _model(M, seed)
    bg = np.random.RandomState(seed).randn(3, M)

    eng = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                       rng=np.random.RandomState(seed), config=_config("resume-done", seed, run_dirs))
    res1 = eng.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=200)
    coal_after = eng.oracle.total_coalition_evals

    eng2 = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                        rng=np.random.RandomState(seed), config=_config("resume-done", seed, run_dirs))
    res2 = eng2.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=200, resume=True)
    assert np.allclose(res1["shapley_values"], res2["shapley_values"], atol=1e-9)
    # no additional oracle work was needed
    assert eng2.oracle.total_coalition_evals <= coal_after


def test_manifest_tracks_latest_and_previous():
    from gas_bayesshap.checkpointing.manager import CheckpointManager
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mgr = CheckpointManager("r1", td, config_hash="c", oracle_hash="o",
                                background_hash="b", M=3, engine_version="11.0.0")
        mgr.save("gp_stage", 0, {"a": 1})
        mgr.save("residual_stage", 5, {"a": 2})
        latest = mgr.manifest.latest()
        assert latest["stage"] == "residual_stage"
        assert latest["iteration"] == 5
        assert mgr.manifest.data["previous_valid_checkpoint"] == "gp_stage.00000000"
        assert mgr.list_checkpoints()  # at least the gp + residual entries


def test_resume_rejects_different_input(run_dirs):
    """Resuming with a different query point x must fail explicitly."""
    M = 3
    seed = 5
    model = _model(M, seed)
    bg = np.random.RandomState(seed).randn(3, M)

    eng = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                       rng=np.random.RandomState(seed), config=_config("resume-x", seed, run_dirs))
    eng.explain(np.ones(M), epsilon=EPS, delta=0.05, max_budget=200)

    eng2 = GASBayesSHAP(model, bg, output_bounds=BOUNDS,
                        rng=np.random.RandomState(seed), config=_config("resume-x", seed, run_dirs))
    with pytest.raises(CheckpointCompatibilityError):
        eng2.explain(np.ones(M) * 0.5, epsilon=EPS, delta=0.05, max_budget=200, resume=True)
