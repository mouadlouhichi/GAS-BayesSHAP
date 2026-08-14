"""Coalition-value cache (spec section 33).

Cache keys incorporate every semantic component:

    oracle hash | input hash | background hash | coalition bitmask | config hash

A cache built with different hashes is **incompatible** and is rejected
(never silently reused).  Cache hits return the stored value without any
query-counter increment; every miss is counted and the value stored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from ..utils.hashing import cache_key, json_sha256
from ..utils.serialization import load_json, write_json_atomic


class CacheCompatibilityError(RuntimeError):
    """Raised when an on-disk cache is incompatible with the current run."""


class CoalitionCache:
    def __init__(
        self,
        config_hash: str,
        oracle_hash: str,
        background_hash: str,
        persist_path: Optional[os.PathLike] = None,
        enabled: bool = True,
    ):
        self.config_hash = str(config_hash)
        self.oracle_hash = str(oracle_hash)
        self.background_hash = str(background_hash)
        self.persist_path = Path(persist_path) if persist_path else None
        self.enabled = bool(enabled)
        self._store: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

        if self.persist_path is not None:
            self._load_or_init()

    # ------------------------------------------------------------------ #
    def key(self, input_h: str, bitmask: int) -> str:
        return cache_key(self.oracle_hash, input_h, self.background_hash, bitmask, self.config_hash)

    def get(self, key: str) -> Optional[float]:
        if not self.enabled:
            return None
        val = self._store.get(key)
        if val is not None:
            self.hits += 1
            return float(val)
        return None

    def put(self, key: str, value: float) -> None:
        if not self.enabled:
            return
        self._store[key] = float(value)
        self.misses += 1  # a put follows a miss in the oracle flow

    def invalidate(self) -> None:
        self._store.clear()
        self.invalidations += 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------ #
    def _manifest(self) -> dict:
        return {
            "config_hash": self.config_hash,
            "oracle_hash": self.oracle_hash,
            "background_hash": self.background_hash,
            "n_entries": len(self._store),
            "entries_hash": json_sha256(
                {k: v for k, v in sorted(self._store.items())}
            ) if self._store else "",
        }

    def _load_or_init(self) -> None:
        if not self.persist_path.exists():
            self._store = {}
            self._write()
            return
        data = load_json(self.persist_path)
        if data.get("config_hash") != self.config_hash:
            raise CacheCompatibilityError(
                "on-disk cache config_hash mismatch — refusing incompatible cache reuse"
            )
        if data.get("oracle_hash") != self.oracle_hash:
            raise CacheCompatibilityError(
                "on-disk cache oracle_hash mismatch — refusing incompatible cache reuse"
            )
        if data.get("background_hash") != self.background_hash:
            raise CacheCompatibilityError(
                "on-disk cache background_hash mismatch — refusing incompatible cache reuse"
            )
        self._store = {k: float(v) for k, v in data.get("entries", {}).items()}

    def _write(self) -> None:
        if self.persist_path is None:
            return
        data = self._manifest()
        data["entries"] = self._store
        write_json_atomic(self.persist_path, data, sort_keys=True)

    def persist(self) -> None:
        if self.persist_path is not None:
            self._write()

    @classmethod
    def empty(cls) -> "CoalitionCache":
        return cls(config_hash="", oracle_hash="", background_hash="", enabled=False)
