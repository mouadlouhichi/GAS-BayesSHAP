"""Run state: query meters, iteration, stage, RNG (spec sections 31 & 38)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..utils.rng_state import dict_to_rng_state, rng_state_to_dict


@dataclass
class RunState:
    run_id: str
    stage: str = "PREFLIGHT"
    iteration: int = 0
    num_coalition_evals: int = 0
    num_model_evals: int = 0
    num_gp_predictions: int = 0
    num_residual_samples: int = 0
    num_sampling_rounds: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    rng_state: Optional[dict] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
        data = asdict(self)
        if rng is not None:
            data["rng_state"] = rng_state_to_dict(rng)
        return data

    def restore(self, data: Dict[str, Any], rng: Optional[np.random.RandomState] = None) -> None:
        for k in ("stage", "iteration", "num_coalition_evals", "num_model_evals",
                  "num_gp_predictions", "num_residual_samples", "num_sampling_rounds",
                  "cache_hits", "cache_misses"):
            if k in data:
                setattr(self, k, data[k])
        if rng is not None and data.get("rng_state") is not None:
            dict_to_rng_state(rng, data["rng_state"])

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
