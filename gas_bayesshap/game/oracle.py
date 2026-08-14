"""Domain-agnostic coalition oracle with exact query accounting.

The oracle is the single, deterministic entry point for evaluating the
interventional expectation ``v(S)``:

* retained features come from the query point ``x``;
* excluded features come from a **frozen empirical background**;
* background values are **not** conditioned on included features
  (interventional / marginal imputation, spec Convention 1).

Query accounting (spec sections 3 & 31):

* ``num_coalition_evals``: every true oracle call ``v(S)`` increments by 1.
  **Cache hits do not count.**
* ``num_model_evals``: ``B`` forward passes for a hybrid coalition,
  ``0`` for the empty coalition (baseline shortcut), ``1`` for the full set.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..cache.coalition_cache import CoalitionCache
from ..utils.hashing import background_hash, cache_key, input_hash, oracle_hash

ModelFn = Callable[[np.ndarray], float]


class CoalitionOracle:
    """Deterministic interventional game oracle.

    Parameters
    ----------
    model_fn:
        ``model_fn(x) -> float`` — single-instance forward pass.
    background:
        ``(B, M)`` array of empirical background samples (frozen).
    output_bounds:
        Optional known global range ``(L, U)`` of ``model_fn``.
    model_tag:
        Optional stable string identifying ``model_fn`` for hashing
        (defaults to ``model_fn.__name__``).
    cache:
        Optional :class:`CoalitionCache`; when enabled, cache hits return the
        stored value without incrementing counters.
    config_hash:
        Configuration hash included in cache keys.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        background: np.ndarray,
        output_bounds: Optional[tuple] = None,
        model_tag: Optional[str] = None,
        cache: Optional[CoalitionCache] = None,
        config_hash: Optional[str] = None,
        logger=None,
    ):
        if not callable(model_fn):
            raise TypeError("model_fn must be callable")
        self.model_fn = model_fn
        self.background = np.array(np.atleast_2d(background), dtype=np.float64, copy=True)
        self.B, self.M = self.background.shape
        if self.B < 1 or self.M < 1:
            raise ValueError("background must have at least 1 row and 1 column")

        self.output_bounds = output_bounds
        if output_bounds is not None:
            L, U = float(output_bounds[0]), float(output_bounds[1])
            if not (np.isfinite(L) and np.isfinite(U) and L < U):
                raise ValueError(f"output_bounds must satisfy -inf < L < U < inf, got {(L, U)}")
            self.output_bounds = (L, U)

        self.model_tag = model_tag if model_tag is not None else getattr(model_fn, "__name__", "model_fn")
        self.cache: Optional[CoalitionCache] = cache
        self.config_hash = config_hash or ""
        self.logger = logger

        # hashes (frozen at construction)
        self.background_h = background_hash(self.background)
        self.oracle_h = oracle_hash(self.model_tag, self.background)

        # query meters
        self.total_coalition_evals = 0
        self.total_model_evals = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Precompute baseline expectation E[f(X)] over the background
        bg_preds = [self.model_fn(self.background[b]) for b in range(self.B)]
        self.total_model_evals += self.B
        self.E_base = float(np.mean(bg_preds))

    # ------------------------------------------------------------------ #
    def _log(self, event: str, **fields) -> None:
        if self.logger is not None and hasattr(self.logger, "event"):
            self.logger.event("oracle", event=event, **fields)

    def evaluate(self, x: np.ndarray, coalition: np.ndarray) -> float:
        """Evaluate ``v(S)`` for coalition mask ``S`` (exact accounting)."""
        x = np.asarray(x, dtype=np.float64)
        S_mask = np.asarray(coalition, dtype=bool)
        if S_mask.shape[0] != self.M:
            raise ValueError(f"coalition length {S_mask.shape[0]} != M={self.M}")

        bitmask = int(np.packbits(S_mask)[0]) if self.M <= 8 else _bitmask(S_mask)
        if self.cache is not None:
            key = cache_key(self.oracle_h, input_hash(x), self.background_h, bitmask, self.config_hash)
            hit = self.cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                self._log("cache_hit", key=key, value=hit)
                return float(hit)

        self.total_coalition_evals += 1
        self.cache_misses += 1
        value = self._evaluate_uncached(x, S_mask)
        self._log(
            "oracle_call",
            coalition=_bitmask(S_mask),
            value=value,
            model_evals=self._last_model_evals,
        )
        if self.cache is not None:
            key = cache_key(self.oracle_h, input_hash(x), self.background_h, bitmask, self.config_hash)
            self.cache.put(key, float(value))
        return float(value)

    # ------------------------------------------------------------------ #
    def _evaluate_uncached(self, x: np.ndarray, S_mask: np.ndarray) -> float:
        """Raw evaluation with the spec's model-eval shortcuts."""
        if np.all(S_mask):
            # full instance: single model pass
            self._last_model_evals = 1
            self.total_model_evals += 1
            return float(self.model_fn(x))
        if not np.any(S_mask):
            # empty coalition: baseline shortcut, zero model passes
            self._last_model_evals = 0
            return float(self.E_base)

        X_hybrid = np.tile(x, (self.B, 1))
        X_hybrid[:, ~S_mask] = self.background[:, ~S_mask]
        preds = [self.model_fn(X_hybrid[b]) for b in range(self.B)]
        self.total_model_evals += self.B
        self._last_model_evals = self.B
        return float(np.mean(preds))

    def evaluate_bitmask(self, x: np.ndarray, bitmask: int) -> float:
        from .subsets import bitmask_to_mask
        return self.evaluate(x, bitmask_to_mask(bitmask, self.M))

    def query_snapshot(self) -> Dict[str, int]:
        """Global (cumulative) query counters."""
        return {
            "num_coalition_evals": self.total_coalition_evals,
            "num_model_evals": self.total_model_evals,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def reset_meters(self) -> None:
        self.total_coalition_evals = 0
        self.total_model_evals = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def validate_determinism(self, x: np.ndarray, coalition: np.ndarray, tol: float = 1e-12) -> bool:
        """Re-evaluate once (bypassing cache) and compare."""
        v1 = self.evaluate(x, coalition)
        v2 = self._evaluate_uncached(np.asarray(x), np.asarray(coalition, dtype=bool))
        return abs(v1 - v2) <= tol


def _bitmask(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool)
    bitmask = 0
    for bit in range(len(m)):
        if m[bit]:
            bitmask |= 1 << bit
    return bitmask
