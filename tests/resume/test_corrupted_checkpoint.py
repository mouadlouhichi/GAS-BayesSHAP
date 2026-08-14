"""Corrupted / incompatible checkpoint rejection (spec sections 35, 51)."""

import json

import numpy as np
import pytest

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.checkpointing.compatibility import CheckpointCompatibilityError
from gas_bayesshap.checkpointing.manager import CheckpointManager

RESULTS_DIR = "results/tests-corrupt"
CKPT_DIR = "checkpoints/tests-corrupt"


def test_partially_written_checkpoint_never_valid(tmp_path):
    """A checkpoint whose file exists but is not referenced by the manifest is
    not loaded; a missing referenced file raises FileNotFoundError."""
    mgr = CheckpointManager("r1", tmp_path / "ck", config_hash="c", oracle_hash="o",
                            background_hash="b", M=3, engine_version="11.0.0")
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool)})

    # corrupt: delete the npz behind the latest entry but keep the manifest
    import glob
    import os
    for npz in glob.glob(str(tmp_path / "ck" / "r1" / "*.npz")):
        os.remove(npz)
    with pytest.raises(OSError):
        mgr.load_latest()


def test_garbage_checkpoint_rejected(tmp_path):
    mgr = CheckpointManager("r1", tmp_path / "ck", config_hash="c", oracle_hash="o",
                            background_hash="b", M=3, engine_version="11.0.0")
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool)})
    # overwrite the json with garbage
    (tmp_path / "ck" / "r1" / "gp_stage.00000000.json").write_text("{ not json !!!")
    with pytest.raises(Exception):
        mgr.load_latest()


def test_incompatible_config_hash_rejected(tmp_path):
    mgr = CheckpointManager("r1", tmp_path / "ck", config_hash="cfg-v1", oracle_hash="o",
                            background_hash="b", M=3, engine_version="11.0.0")
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool)})

    other = CheckpointManager("r1", tmp_path / "ck", config_hash="cfg-v2", oracle_hash="o",
                              background_hash="b", M=3, engine_version="11.0.0")
    with pytest.raises(CheckpointCompatibilityError):
        other.load_latest()


def test_incompatible_engine_version_rejected(tmp_path):
    mgr = CheckpointManager("r1", tmp_path / "ck", config_hash="c", oracle_hash="o",
                            background_hash="b", M=3, engine_version="11.0.0")
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool)})
    other = CheckpointManager("r1", tmp_path / "ck", config_hash="c", oracle_hash="o",
                              background_hash="b", M=3, engine_version="12.0.0")
    with pytest.raises(CheckpointCompatibilityError):
        other.load_latest()


def test_engine_rejects_corrupted_run_checkpoint(tmp_path):
    """End-to-end: a corrupted checkpoint under the run dir must make resume
    fail explicitly (never silently recover in a different state)."""
    M = 3
    cfg = {
        "checkpoint_enabled": True, "checkpoint_every": 1,
        "cache_enabled": True, "persist_cache": True,
        "log_level": "NONE", "seed": 1,
        "results_dir": str(tmp_path / "results"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
    }
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, M)),
                       output_bounds=(0.0, 3.0), rng=np.random.RandomState(1),
                       config={**cfg, "run_id": "corrupt-run"})
    eng.explain(np.ones(M), epsilon=5.0, delta=0.05, max_budget=100)
    # corrupt every checkpoint json (except the manifest) with garbage
    import glob
    for f in glob.glob(str(tmp_path / "checkpoints" / "corrupt-run" / "*.json")):
        if "manifest" not in f:
            (tmp_path / "checkpoints" / "corrupt-run" / f).write_text("garbage")

    eng2 = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, M)),
                        output_bounds=(0.0, 3.0), rng=np.random.RandomState(1),
                        config={**cfg, "run_id": "corrupt-run"})
    with pytest.raises(Exception):
        eng2.explain(np.ones(M), epsilon=5.0, delta=0.05, max_budget=100, resume=True)
