"""Checkpoint integrity verification and fallback
(audit finding: High 2)."""

import os

import numpy as np
import pytest

from gas_bayesshap.checkpointing.compatibility import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
)
from gas_bayesshap.checkpointing.manager import CheckpointManager


def _manager(tmp_path, engine_version="11.0.0"):
    return CheckpointManager("r1", tmp_path / "ck", config_hash="c", oracle_hash="o",
                             background_hash="b", M=3, engine_version=engine_version)


def test_payload_tampering_detected(tmp_path):
    """Byte-level corruption of the npz must be detected on load."""
    mgr = _manager(tmp_path)
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.ones(3)})
    # tamper with the only checkpoint's npz bytes (no fallback available)
    latest = mgr.manifest.latest()["name"]
    npz_path = tmp_path / "ck" / "r1" / f"{latest}.npz"
    data = bytearray(npz_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    npz_path.write_bytes(bytes(data))

    with pytest.raises(CheckpointIntegrityError):
        mgr.load_latest()


def test_fallback_to_previous_valid_checkpoint(tmp_path):
    """When the latest checkpoint is corrupted, load_latest must fall back to
    the previous valid one."""
    mgr = _manager(tmp_path)
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.zeros(3)})
    mgr.save("residual_stage", 1, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.ones(3)})

    # corrupt the latest (residual_stage.1)
    latest = mgr.manifest.latest()["name"]
    npz_path = tmp_path / "ck" / "r1" / f"{latest}.npz"
    data = bytearray(npz_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    npz_path.write_bytes(bytes(data))

    state = mgr.load_latest()  # must fall back to gp_stage.0
    assert state["stage"] == "gp_stage"
    assert np.allclose(state["alpha"], np.zeros(3))


def test_previous_also_corrupted_raises(tmp_path):
    mgr = _manager(tmp_path)
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.zeros(3)})
    mgr.save("residual_stage", 1, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.ones(3)})
    # corrupt BOTH checkpoints
    for name in ("gp_stage.00000000", "residual_stage.00000001"):
        npz_path = tmp_path / "ck" / "r1" / f"{name}.npz"
        data = bytearray(npz_path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        npz_path.write_bytes(bytes(data))
    with pytest.raises(CheckpointIntegrityError):
        mgr.load_latest()


def test_intact_checkpoint_loads():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        mgr = _manager(Path(td))
        mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.array([1.0, 2.0, 3.0])})
        state = mgr.load_latest()
        assert state["stage"] == "gp_stage"
        assert np.allclose(state["alpha"], [1.0, 2.0, 3.0])


def test_compatibility_mismatch_not_treated_as_corruption(tmp_path):
    mgr = _manager(tmp_path)
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.zeros(3)})
    other = CheckpointManager("r1", tmp_path / "ck", config_hash="DIFFERENT", oracle_hash="o",
                              background_hash="b", M=3, engine_version="11.0.0")
    with pytest.raises(CheckpointCompatibilityError):
        other.load_latest()
