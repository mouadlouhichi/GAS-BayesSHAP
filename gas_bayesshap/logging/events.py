"""JSONL event logging (spec section 37).

Writes one JSON object per line into per-topic files under
``results/runs/<run_id>/logs/``:

``run.log, events.jsonl, oracle_calls.jsonl, gp_updates.jsonl,
acquisition.jsonl, residual_sampling.jsonl, neyman.jsonl,
certification.jsonl, checkpoints.jsonl, errors.log``

Every event carries ``timestamp, run_id, stage, iteration, event, status,
num_coalition_evals, num_model_evals``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.serialization import ensure_dir


class EventLogger:
    FILES = (
        "events",
        "oracle_calls",
        "gp_updates",
        "acquisition",
        "residual_sampling",
        "neyman",
        "certification",
        "checkpoints",
    )

    def __init__(
        self,
        run_id: str,
        log_dir: os.PathLike,
        stage: str = "PREFLIGHT",
        counters: Optional[Dict[str, int]] = None,
    ):
        self.run_id = str(run_id)
        self.log_dir = ensure_dir(log_dir)
        self.stage = stage
        self.counters = dict(counters or {})
        self._handles: Dict[str, Any] = {}
        for name in self.FILES:
            self._handles[name] = open(self.log_dir / f"{name}.jsonl", "a", encoding="utf-8")

    # ------------------------------------------------------------------ #
    def _base(self, event: str, status: str = "") -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "stage": self.stage,
            "iteration": self.counters.get("iteration", None),
            "event": event,
            "status": status,
            "num_coalition_evals": self.counters.get("num_coalition_evals", 0),
            "num_model_evals": self.counters.get("num_model_evals", 0),
        }

    def event(self, topic: str, event: str, status: str = "", **fields: Any) -> None:
        """Log to a topic file (and mirror core events to ``events.jsonl``)."""
        if topic not in self._handles:
            topic = "events"
        rec = self._base(event, status)
        rec.update(fields)
        line = json.dumps(rec, default=_json_default, sort_keys=True)
        self._handles[topic].write(line + "\n")
        self._handles[topic].flush()
        if topic != "events":
            self._handles["events"].write(line + "\n")
            self._handles["events"].flush()

    def error(self, message: str, **fields: Any) -> None:
        rec = self._base("error", "ERROR")
        rec["message"] = message
        rec.update(fields)
        with open(self.log_dir / "errors.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=_json_default, sort_keys=True) + "\n")

    def set_counters(self, **counters: int) -> None:
        self.counters.update(counters)

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _json_default(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)
