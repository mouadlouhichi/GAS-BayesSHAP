"""Structured logging layer (spec section 37)."""

import json
import os

import numpy as np

from gas_bayesshap import GASBayesSHAP

LOG_FILES = [
    "run.log", "events.jsonl", "oracle_calls.jsonl", "gp_updates.jsonl",
    "acquisition.jsonl", "residual_sampling.jsonl", "neyman.jsonl",
    "certification.jsonl", "checkpoints.jsonl", "errors.log",
]


def _run(tmp_path, run_id="logtest"):
    M = 4
    rng = np.random.RandomState(0)
    w = rng.randn(M)

    def model(x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, w) / np.sqrt(M)))

    eng = GASBayesSHAP(
        model, rng.randn(5, M), output_bounds=(0.0, 1.0),
        rng=np.random.RandomState(0),
        config={
            "checkpoint_enabled": True, "cache_enabled": True, "persist_cache": True,
            "log_level": "INFO", "seed": 0,
            "results_dir": str(tmp_path / "results"),
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "run_id": run_id,
        },
    )
    return eng.explain(np.ones(M), epsilon=1.0, delta=0.05, max_budget=40,
                       n_pilot=2, n_active_steps=5)


def test_all_log_files_created(tmp_path):
    _run(tmp_path)
    log_dir = tmp_path / "results" / "logtest" / "logs"
    assert log_dir.is_dir()
    for name in LOG_FILES:
        assert (log_dir / name).exists(), f"missing log file {name}"


def test_topic_files_populated(tmp_path):
    _run(tmp_path)
    log_dir = tmp_path / "results" / "logtest" / "logs"

    # per-topic JSONL files must contain real events (not empty)
    for name in ["events", "oracle_calls", "gp_updates", "acquisition",
                 "residual_sampling", "neyman", "certification", "checkpoints"]:
        p = log_dir / f"{name}.jsonl"
        assert p.stat().st_size > 0, f"{name}.jsonl is empty"

    # run.log must contain stage events
    run_log = (log_dir / "run.log").read_text()
    assert "PREFLIGHT" in run_log and "FINAL_RESULT" in run_log

    # oracle_calls records have the required envelope fields
    first = json.loads((log_dir / "oracle_calls.jsonl").read_text().splitlines()[0])
    for key in ("timestamp", "run_id", "stage", "iteration", "event",
                "status", "num_coalition_evals", "num_model_evals"):
        assert key in first, f"missing envelope key {key}"

    # certification file contains the full width vector events
    cert = (log_dir / "certification.jsonl").read_text()
    assert "width_check" in cert or "budget_exhausted" in cert


def test_events_have_counters_and_stage(tmp_path):
    _run(tmp_path)
    log_dir = tmp_path / "results" / "logtest" / "logs"
    stages = {
        "PREFLIGHT", "GP_INITIALIZATION", "ACTIVE_GP", "BOUNDED_SURROGATE",
        "SURROGATE_SHAPLEY", "RESIDUAL_PILOT", "NEYMAN_ALLOCATION",
        "ADAPTIVE_CERTIFICATION", "EFFICIENCY_PROJECTION", "FINAL_RESULT",
        "gp_stage", "residual_stage", "certification_stage", "final_stage",
    }
    for line in (log_dir / "events.jsonl").read_text().splitlines():
        rec = json.loads(line)
        assert rec["run_id"] == "logtest"
        assert "num_coalition_evals" in rec
        assert "num_model_evals" in rec
        assert rec["stage"] in stages, f"unexpected stage {rec['stage']!r}"


def test_checkpoint_events_logged(tmp_path):
    _run(tmp_path)
    log_dir = tmp_path / "results" / "logtest" / "logs"
    text = (log_dir / "checkpoints.jsonl").read_text()
    assert "checkpoint_saved" in text


def test_log_level_none_disables_logs(tmp_path):
    M = 3
    rng = np.random.RandomState(0)

    eng = GASBayesSHAP(
        lambda x: float(np.sum(x)), np.zeros((2, M)), output_bounds=(0.0, 3.0),
        rng=rng,
        config={
            "checkpoint_enabled": False, "cache_enabled": False,
            "log_level": "NONE", "results_dir": str(tmp_path / "r"),
            "checkpoints_dir": str(tmp_path / "c"), "run_id": "quiet",
        },
    )
    eng.explain(np.ones(M), epsilon=1.0, max_budget=30)
    assert not (tmp_path / "r" / "quiet" / "logs").exists()
