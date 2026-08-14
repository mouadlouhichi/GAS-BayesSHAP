"""Checkpoint manager: atomic save/load/resume (spec sections 34-36).

Layout::

    checkpoints/<run_id>/
        gp_stage.checkpoint.npz     (arrays)
        gp_stage.checkpoint.json    (metadata + small state)
        residual_stage.<iter>.npz
        residual_stage.<iter>.json
        certification_stage.npz / .json
        final_stage.npz / .json
        checkpoint_manifest.json

Every write is atomic (temp file -> flush -> fsync -> rename) so a partially
written checkpoint is never valid.  ``--resume`` loads the latest valid
checkpoint and restores GP state, residual observations, Neyman state, RNG
state, query counters and iteration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.serialization import (
    ensure_dir,
    load_json,
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from .compatibility import verify_compatibility
from .manifest import CheckpointManifest

CHECKPOINT_STAGES = ("gp_stage", "residual_stage", "certification_stage", "final_stage")


class CheckpointManager:
    def __init__(
        self,
        run_id: str,
        directory: os.PathLike,
        config_hash: str,
        oracle_hash: str,
        background_hash: str,
        M: int,
        engine_version: str,
        logger=None,
    ):
        self.run_id = str(run_id)
        self.directory = ensure_dir(Path(directory) / self.run_id)
        self.config_hash = config_hash
        self.oracle_hash = oracle_hash
        self.background_hash = background_hash
        self.M = int(M)
        self.engine_version = engine_version
        self.logger = logger
        self.manifest = CheckpointManifest(self.directory)

    # ------------------------------------------------------------------ #
    def _log(self, event: str, **fields) -> None:
        if self.logger is not None and hasattr(self.logger, "event"):
            self.logger.event("checkpoints", event=event, **fields)

    def save(self, stage: str, iteration: int, payload: Dict[str, Any]) -> Path:
        """Atomically persist a checkpoint and update the manifest."""
        if stage not in CHECKPOINT_STAGES:
            raise ValueError(f"unknown checkpoint stage {stage!r}")
        payload = dict(payload)
        payload.setdefault("config_hash", self.config_hash)
        payload.setdefault("oracle_hash", self.oracle_hash)
        payload.setdefault("background_hash", self.background_hash)
        payload.setdefault("M", self.M)
        payload.setdefault("engine_version", self.engine_version)
        payload.setdefault("stage", stage)
        payload.setdefault("iteration", int(iteration))
        payload.setdefault("run_id", self.run_id)

        name = f"{stage}.{int(iteration):08d}"
        npz_path = self.directory / f"{name}.npz"
        json_path = self.directory / f"{name}.json"

        arrays = {}
        meta = {}
        for k, v in payload.items():
            if _is_array(v):
                arrays[k] = v
            else:
                meta[k] = v

        write_npz_atomic(npz_path, **arrays)
        meta["_npz"] = npz_path.name
        write_json_atomic(json_path, meta, sort_keys=True)

        # payload hash over the full state (arrays + meta)
        from .manifest import _jsonable_payload
        payload_hash = self.manifest.recompute_payload_hash(payload)
        self.manifest.update(
            checkpoint_name=name,
            stage=stage,
            iteration=int(iteration),
            query_count=int(meta.get("num_coalition_evals", 0)),
            config_hash=self.config_hash,
            oracle_hash=self.oracle_hash,
            payload_hash=payload_hash,
        )
        self._log("checkpoint_saved", checkpoint=name, stage=stage, iteration=int(iteration))
        return json_path

    # ------------------------------------------------------------------ #
    def load_latest(self) -> Dict[str, Any]:
        """Load the latest valid checkpoint (raises if none exists)."""
        latest = self.manifest.latest()
        if latest is None:
            raise FileNotFoundError(f"no valid checkpoint for run {self.run_id}")
        name = latest["name"]
        return self.load(name)

    def load(self, name: str) -> Dict[str, Any]:
        json_path = self.directory / f"{name}.json"
        npz_name = load_json(json_path).get("_npz")
        if npz_name is None:
            raise FileNotFoundError(f"checkpoint {name} missing npz reference")
        npz_path = self.directory / npz_name
        npz = load_npz(npz_path)
        meta = load_json(json_path)
        state = dict(meta)
        for k in npz.files:
            state[k] = npz[k]
        verify_compatibility(
            state,
            config_hash=self.config_hash,
            oracle_hash=self.oracle_hash,
            background_hash=self.background_hash,
            M=self.M,
            engine_version=self.engine_version,
        )
        self._log("checkpoint_restored", checkpoint=name, stage=state.get("stage"))
        return state

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        out = []
        for p in sorted(self.directory.glob("*.json")):
            if p.name == "checkpoint_manifest.json":
                continue
            try:
                meta = load_json(p)
                out.append(
                    {
                        "name": p.stem,
                        "stage": meta.get("stage"),
                        "iteration": meta.get("iteration"),
                        "run_id": meta.get("run_id"),
                    }
                )
            except Exception:
                continue
        return out

    def latest_stage_and_iteration(self) -> Optional[Dict[str, Any]]:
        return self.manifest.latest()

    def manifest_dict(self) -> Dict[str, Any]:
        return self.manifest.to_dict()


def _is_array(v: Any) -> bool:
    import numpy as np
    return isinstance(v, np.ndarray)
