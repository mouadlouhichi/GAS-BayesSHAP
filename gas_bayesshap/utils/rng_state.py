"""JSON-safe conversion of ``numpy.random.RandomState`` state."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def rng_state_to_dict(rng: np.random.RandomState) -> Dict[str, Any]:
    """``get_state()`` tuple -> JSON-safe dict (ndarray converted to list)."""
    st = rng.get_state()  # (str, ndarray(uint32), int, int, float)
    return {
        "bit_generator": str(st[0]),
        "keys": [int(k) for k in st[1]],
        "pos": int(st[2]),
        "has_gauss": int(st[3]),
        "cached_gaussian": float(st[4]),
    }


def dict_to_rng_state(rng: np.random.RandomState, state: Dict[str, Any]) -> np.random.RandomState:
    """Restore a RandomState from a JSON-safe dict."""
    st = (
        state["bit_generator"],
        np.array(state["keys"], dtype=np.uint32),
        int(state["pos"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )
    rng.set_state(st)
    return rng


def rng_state_hash(rng: np.random.RandomState) -> str:
    from .hashing import stable_hash
    return stable_hash(rng_state_to_dict(rng), namespace="run")
