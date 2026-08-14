"""Checkpoint manifest (spec section 35).

``checkpoint_manifest.json`` records, for every valid checkpoint:

``latest_valid_checkpoint, previous_valid_checkpoint, stage, iteration,
query_count, config_hash, oracle_hash, result_hash``.

A partially written checkpoint is never referenced by the manifest (writes
are atomic and the manifest is updated only after a successful save).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.hashing import json_sha256
from ..utils.serialization import load_json, write_json_atomic


class CheckpointManifest:
    def __init__(self, directory: os.PathLike):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "checkpoint_manifest.json"
        self.data: Dict[str, Any] = self._load_or_init()

    def _load_or_init(self) -> Dict[str, Any]:
        if self.path.exists():
            data = load_json(self.path)
            if isinstance(data, dict):
                data.setdefault("checkpoint_hashes", {})
                data.setdefault("background_hash", "")
                data.setdefault("engine_version", "")
                return data
        return {
            "latest_valid_checkpoint": None,
            "previous_valid_checkpoint": None,
            "stage": None,
            "iteration": None,
            "query_count": 0,
            "config_hash": "",
            "oracle_hash": "",
            "background_hash": "",
            "engine_version": "",
            "result_hash": "",
            "checkpoint_hashes": {},
        }

    def update(
        self,
        checkpoint_name: str,
        stage: str,
        iteration: int,
        query_count: int,
        config_hash: str,
        oracle_hash: str,
        payload_hash: str,
        background_hash: str = "",
        engine_version: str = "",
        npz_sha256: str = "",
        json_sha256: str = "",
    ) -> None:
        previous = self.data.get("latest_valid_checkpoint")
        hashes = dict(self.data.get("checkpoint_hashes", {}))
        hashes[str(checkpoint_name)] = {
            "payload": str(payload_hash),
            "npz": str(npz_sha256),
            "json": str(json_sha256),
        }
        self.data = {
            "latest_valid_checkpoint": str(checkpoint_name),
            "previous_valid_checkpoint": previous,
            "stage": stage,
            "iteration": int(iteration),
            "query_count": int(query_count),
            "config_hash": str(config_hash),
            "oracle_hash": str(oracle_hash),
            "background_hash": str(background_hash),
            "engine_version": str(engine_version),
            "result_hash": str(payload_hash),
            "checkpoint_hashes": hashes,
        }
        self._persist()

    def _persist(self) -> None:
        write_json_atomic(self.path, self.data, sort_keys=True)

    def latest(self) -> Optional[Dict[str, Any]]:
        name = self.data.get("latest_valid_checkpoint")
        if not name:
            return None
        out = {k: v for k, v in self.data.items() if k != "latest_valid_checkpoint"}
        out["name"] = name
        return out

    def integrity_record(self, checkpoint_name: str) -> Optional[Dict[str, str]]:
        """Per-checkpoint integrity record (payload/npz/json hashes)."""
        return self.data.get("checkpoint_hashes", {}).get(str(checkpoint_name))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    def recompute_payload_hash(self, payload: Dict[str, Any]) -> str:
        return json_sha256(_jsonable_payload(payload))


def _jsonable_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    out = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray):
            out[k] = {"dtype": v.dtype.str, "shape": list(v.shape), "data": v.tolist()}
        elif isinstance(v, dict):
            out[k] = _jsonable_payload(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_jsonable_payload(x) if isinstance(x, dict) else x for x in v]
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out
