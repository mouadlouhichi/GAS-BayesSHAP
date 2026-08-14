"""Optional GP-prediction cache.

Disabled by default (``gp_prediction_cache: false``) so the engine's exact
behaviour matches the spec reference; when enabled, identical coalition
predictions return cached values (pure lookups, no query counters involved).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..utils.hashing import stable_hash


class GPPredictionCache:
    def __init__(self, enabled: bool = False, design_hash: str = ""):
        self.enabled = bool(enabled)
        self.design_hash = str(design_hash)
        self._store: Dict[str, float] = {}
        self.hits = 0

    def _key(self, S: np.ndarray) -> str:
        return stable_hash(
            {"design": self.design_hash, "S": np.asarray(S, dtype=bool).tolist()},
            namespace="cache",
        )

    def get(self, S: np.ndarray) -> Optional[float]:
        if not self.enabled:
            return None
        key = self._key(S)
        val = self._store.get(key)
        if val is not None:
            self.hits += 1
        return val

    def put(self, S: np.ndarray, value: float) -> None:
        if not self.enabled:
            return
        self._store[self._key(S)] = float(value)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
