"""Malformed-JSON / missing-file checkpoints must still trigger fallback
(audit #2 Medium caveat)."""

import numpy as np
import pytest

from gas_bayesshap.checkpointing.compatibility import CheckpointIntegrityError
from gas_bayesshap.checkpointing.manager import CheckpointManager


def _mgr(tmp_path):
    return CheckpointManager("r1", tmp_path / "ck", config_hash="c", oracle_hash="o",
                             background_hash="b", M=3, engine_version="11.0.0")


def test_malformed_json_falls_back(tmp_path):
    """A truncated/unparseable latest JSON is an integrity failure, not a
    JSONDecodeError — load_latest() must fall back to the previous checkpoint."""
    mgr = _mgr(tmp_path)
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.zeros(3)})
    mgr.save("residual_stage", 1, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.ones(3)})

    latest = mgr.manifest.latest()["name"]
    (tmp_path / "ck" / "r1" / f"{latest}.json").write_text("{ not json !!!")

    state = mgr.load_latest()  # must fall back to gp_stage.0
    assert state["stage"] == "gp_stage"
    assert np.allclose(state["alpha"], np.zeros(3))


def test_missing_npz_falls_back(tmp_path):
    """A missing NPZ file is an integrity failure (not a bare FileNotFound),
    so the fallback engages."""
    mgr = _mgr(tmp_path)
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.zeros(3)})
    mgr.save("residual_stage", 1, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.ones(3)})

    latest = mgr.manifest.latest()["name"]
    (tmp_path / "ck" / "r1" / f"{latest}.npz").unlink()

    state = mgr.load_latest()
    assert state["stage"] == "gp_stage"


def test_both_corrupt_still_raises(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.save("gp_stage", 0, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.zeros(3)})
    mgr.save("residual_stage", 1, {"D_coalitions": np.eye(3, dtype=bool), "alpha": np.ones(3)})
    for name in ("gp_stage.00000000", "residual_stage.00000001"):
        (tmp_path / "ck" / "r1" / f"{name}.json").write_text("garbage")
    with pytest.raises(CheckpointIntegrityError):
        mgr.load_latest()
