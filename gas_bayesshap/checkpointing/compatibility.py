"""Checkpoint/cache compatibility verification (spec sections 33, 36, 51).

A checkpoint or cache is reusable only if the run identity components match:

* configuration hash (``config_hash``),
* oracle hash (``oracle_hash``),
* background hash (``background_hash``),
* feature count ``M``,
* package/engine version.

Never silently reuse an incompatible state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint is incompatible with the current run."""


def verify_compatibility(
    state: Dict[str, Any],
    config_hash: str,
    oracle_hash: str,
    background_hash: str,
    M: int,
    engine_version: str,
) -> None:
    """Raise :class:`CheckpointCompatibilityError` on any mismatch."""
    mismatches = []
    if state.get("config_hash") != config_hash:
        mismatches.append(
            f"config_hash: checkpoint={state.get('config_hash')!r} run={config_hash!r}"
        )
    if state.get("oracle_hash") != oracle_hash:
        mismatches.append(
            f"oracle_hash: checkpoint={state.get('oracle_hash')!r} run={oracle_hash!r}"
        )
    if state.get("background_hash") != background_hash:
        mismatches.append(
            f"background_hash: checkpoint={state.get('background_hash')!r} "
            f"run={background_hash!r}"
        )
    if state.get("M") != M:
        mismatches.append(f"M: checkpoint={state.get('M')!r} run={M!r}")
    if state.get("engine_version") != engine_version:
        mismatches.append(
            f"engine_version: checkpoint={state.get('engine_version')!r} "
            f"run={engine_version!r}"
        )
    if mismatches:
        raise CheckpointCompatibilityError(
            "incompatible checkpoint: " + "; ".join(mismatches)
        )
