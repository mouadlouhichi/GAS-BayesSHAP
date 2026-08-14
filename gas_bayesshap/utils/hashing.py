"""Stable, dependency-free hashing helpers used across the engine.

Hashes are computed with domain separation so that different semantic objects
(cache keys, configs, oracles, backgrounds, inputs) can never collide
accidentally.  All hashes are hex digests of SHA-256.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any, Mapping, Optional

import numpy as np

_NAMESPACES = {
    "config": b"gasbs.config.v1",
    "oracle": b"gasbs.oracle.v1",
    "background": b"gasbs.background.v1",
    "input": b"gasbs.input.v1",
    "coalition": b"gasbs.coalition.v1",
    "cache": b"gasbs.cache.v1",
    "run": b"gasbs.run.v1",
    "manifest": b"gasbs.manifest.v1",
}


def _sha256() -> "hashlib._Hash":
    return hashlib.sha256()


def _update_bytes(h: "hashlib._Hash", data: bytes) -> None:
    h.update(data)


def _update_value(h: "hashlib._Hash", value: Any) -> None:
    """Recursively feed a Python/numpy value into the hash."""
    if value is None or isinstance(value, (bool, int, float, str)):
        _update_bytes(h, repr(value).encode("utf-8"))
    elif isinstance(value, bytes):
        _update_bytes(h, value)
    elif isinstance(value, np.ndarray):
        _update_bytes(h, value.dtype.str.encode("utf-8"))
        _update_bytes(h, repr(value.shape).encode("utf-8"))
        # byte-exact payload (order + values)
        _update_bytes(h, value.tobytes(order="C"))
    elif isinstance(value, (list, tuple)):
        _update_bytes(h, b"[")
        for v in value:
            _update_value(h, v)
        _update_bytes(h, b"]")
    elif isinstance(value, Mapping):
        for k in sorted(value.keys(), key=repr):
            _update_bytes(h, b"{")
            _update_value(h, k)
            _update_bytes(h, b":")
            _update_value(h, value[k])
            _update_bytes(h, b"}")
    elif hasattr(value, "get_state"):  # numpy RandomState state tuple
        _update_bytes(h, repr(value.get_state()).encode("utf-8"))
    else:
        _update_bytes(h, repr(value).encode("utf-8"))


def stable_hash(value: Any, namespace: Optional[str] = None) -> str:
    """Return a deterministic SHA-256 hex digest of ``value``.

    Parameters
    ----------
    value:
        Anything JSON/NumPy-serializable.
    namespace:
        Optional domain-separation tag (one of ``_NAMESPACES`` keys or a raw
        string; raw strings are prefixed with ``gasbs.custom.``).
    """
    h = _sha256()
    if namespace is not None:
        ns = _NAMESPACES.get(namespace, b"gasbs.custom." + str(namespace).encode("utf-8"))
        _update_bytes(h, ns)
    _update_value(h, value)
    return h.hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash of the effective run configuration (after default merge)."""
    return stable_hash(dict(config), namespace="config")


def oracle_hash(
    model_tag: str,
    background: np.ndarray,
    artifact_hash: Optional[str] = None,
) -> str:
    """Hash identifying a coalition oracle: model tag + optional artifact
    hash + frozen background.

    ``artifact_hash`` is a caller-supplied digest of the model artifact
    (e.g. ``sha256`` of fitted parameters / ``state_dict`` / model file).
    When it is provided, the identity includes it — two models that differ
    only by parameters are then never cache/checkpoint-compatible.
    """
    return stable_hash(
        {"model_tag": model_tag, "artifact": artifact_hash, "background": np.asarray(background)},
        namespace="oracle",
    )


def background_hash(background: np.ndarray) -> str:
    return stable_hash(np.asarray(background), namespace="background")


def input_hash(x: np.ndarray) -> str:
    return stable_hash(np.asarray(x), namespace="input")


def coalition_hash(bitmask: int, M: int) -> str:
    return stable_hash({"mask": int(bitmask), "M": int(M)}, namespace="coalition")


def cache_key(
    oracle_h: str,
    input_h: str,
    background_h: str,
    bitmask: int,
    config_h: str,
) -> str:
    """Full cache key: every semantic component is represented."""
    return stable_hash(
        {
            "oracle": oracle_h,
            "input": input_h,
            "background": background_h,
            "coalition": int(bitmask),
            "config": config_h,
        },
        namespace="cache",
    )


def json_sha256(obj: Any) -> str:
    """Hash of a JSON-serializable object (used for manifests)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256().update(payload) or hashlib.sha256(payload).hexdigest()


def np_to_json(value: np.ndarray) -> dict:
    """Serialize a NumPy array for JSON persistence."""
    return {"dtype": value.dtype.str, "shape": list(value.shape), "data": value.tolist()}


def json_to_np(value: dict) -> np.ndarray:
    return np.array(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
