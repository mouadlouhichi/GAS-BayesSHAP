"""Configuration loading and validation.

Settings are read from YAML files, merged over defaults, validated against a
schema, and exposed through a lightweight ``Config`` object.  Nothing in the
engine hard-codes experiment settings.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    # --- kernel / surrogate ---
    "sigma0": 1.0,
    "lengthscale": 1.5,
    "eta": 1e-4,
    # --- certification ---
    "epsilon": 0.02,
    "delta": 0.05,
    "max_budget": 1500,        # max individual coalition evaluations in Stage 2
    "max_rounds": None,        # cap on Stage-2 sampling iterations (None = unlimited)
    "n_pilot": 3,
    "n_active_steps": 25,
    "pool_size": None,         # None -> max(32, 2*M)
    "neyman_refresh_interval": None,  # None -> 5*M
    "certification_mode": "STRICT",   # STRICT | PERMISSIVE (PERMISSIVE logs approximate fallback)
    # --- engineering ---
    "seed": 42,
    "output_bounds": None,     # [L, U] or null (-> heuristic bounds flagged)
    "domain_game": "membership",
    "dataset": None,
    "M": None,                 # optional override (group-lag games)
    "checkpoint_every": 1,     # checkpoint every N Stage-2 iterations
    "checkpoint_enabled": True,
    "cache_enabled": True,
    "gp_prediction_cache": False,
    "persist_cache": True,
    "log_level": "INFO",
    "run_id": None,
    "results_dir": "results/runs",
    "checkpoints_dir": "checkpoints",
    "n_trials": 30,            # coverage validation trials
    "numerical_tol": 1e-10,
    "finite_check": True,
    # opt-in empirical residual-range tightening (review task #2)
    # "spec" | "empirical_max" | "holdout"  (non-spec => heuristic bounds)
    "range_mode": "spec",
    "range_safety_factor": 2.0,
    # --- validation hooks ---
    "oracle_validation": False,
    "mathematical_validation": False,
    "validate_boundedness": False,
    "boundedness_sweep_max_M": 12,
}


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


def _validate_type(name: str, value: Any, expected: type) -> None:
    if value is not None and not isinstance(value, expected):
        raise ConfigError(
            f"config key '{name}' must be {expected.__name__}, got {type(value).__name__}"
        )


def validate_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a (merged) configuration dictionary.

    Raises
    ------
    ConfigError
        If any setting is missing, mistyped or out of range.
    """
    for key, default in DEFAULTS.items():
        if key not in cfg:
            raise ConfigError(f"config key '{key}' missing (default {default!r})")

    cfg = dict(cfg)
    _validate_type("sigma0", cfg["sigma0"], (int, float))
    _validate_type("lengthscale", cfg["lengthscale"], (int, float))
    _validate_type("eta", cfg["eta"], (int, float))
    _validate_type("epsilon", cfg["epsilon"], (int, float))
    _validate_type("delta", cfg["delta"], (int, float))
    _validate_type("max_budget", cfg["max_budget"], int)
    _validate_type("max_rounds", cfg["max_rounds"], int)
    _validate_type("n_pilot", cfg["n_pilot"], int)
    _validate_type("n_active_steps", cfg["n_active_steps"], int)
    _validate_type("pool_size", cfg["pool_size"], int)
    _validate_type("neyman_refresh_interval", cfg["neyman_refresh_interval"], int)
    _validate_type("seed", cfg["seed"], int)
    _validate_type("checkpoint_every", cfg["checkpoint_every"], int)
    _validate_type("n_trials", cfg["n_trials"], int)

    if not (cfg["sigma0"] > 0):
        raise ConfigError("sigma0 must be > 0")
    if not (cfg["lengthscale"] > 0):
        raise ConfigError("lengthscale must be > 0")
    if not (0 < cfg["eta"]):
        raise ConfigError("eta (jitter) must be > 0")
    if not (cfg["epsilon"] > 0):
        raise ConfigError("epsilon must be > 0")
    if not (0 < cfg["delta"] < 1):
        raise ConfigError("delta must lie in (0, 1)")
    if cfg["max_budget"] <= 0:
        raise ConfigError("max_budget must be > 0")
    if cfg["n_pilot"] < 0:
        raise ConfigError("n_pilot must be >= 0")
    if cfg["n_active_steps"] < 0:
        raise ConfigError("n_active_steps must be >= 0")
    if cfg["certification_mode"] not in ("STRICT", "PERMISSIVE"):
        raise ConfigError("certification_mode must be 'STRICT' or 'PERMISSIVE'")

    bounds = cfg.get("output_bounds")
    if bounds is not None:
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            raise ConfigError("output_bounds must be [L, U] or null")
        L, U = float(bounds[0]), float(bounds[1])
        if not (L < U) or not np_finite(L, U):
            raise ConfigError(f"output_bounds must satisfy -inf < L < U < inf, got ({L}, {U})")
        cfg["output_bounds"] = (L, U)
    return cfg


def np_finite(*values):
    import math
    return all(math.isfinite(v) for v in values)


def load_config(path: Optional[os.PathLike] = None, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config file, merge over defaults and apply overrides.

    A flat YAML mapping of config keys is expected.  Files with nested
    top-level mappings (preset libraries such as ``configs/games.yaml`` or
    ``configs/experiments.yaml``) are rejected with a helpful message; use
    :func:`load_game_preset` to apply a named game preset.
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config file {p} must contain a YAML mapping")
        for k, v in loaded.items():
            if k not in DEFAULTS:
                if isinstance(v, (dict, list)):
                    raise ConfigError(
                        f"config file {p} looks like a preset library (key '{k}' "
                        f"is a {'mapping' if isinstance(v, dict) else 'list'}), "
                        f"not a flat run config; load a named preset with "
                        f"load_game_preset('{k}') or use a flat YAML"
                    )
                raise ConfigError(f"unknown config key '{k}' in {p}")
            cfg[k] = v
    if overrides:
        for k, v in overrides.items():
            if k not in DEFAULTS:
                raise ConfigError(f"unknown config override '{k}'")
            cfg[k] = v
    return validate_config(cfg)


def load_game_preset(
    game: str,
    base: Optional[Mapping[str, Any]] = None,
    presets_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Load a named domain-game preset from ``configs/games.yaml``.

    Returns the base config (defaults, or ``base`` if given) with the preset
    keys merged in — e.g. the ``membership`` preset sets
    ``domain_game: membership`` and ``output_bounds: [0, 1]`` so runs are
    rigorous by default.
    """
    path = Path(presets_path) if presets_path else Path(__file__).resolve().parents[2] / "configs" / "games.yaml"
    if not path.exists():
        raise ConfigError(f"game presets file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        presets = yaml.safe_load(fh) or {}
    if not isinstance(presets, dict) or game not in presets:
        raise ConfigError(f"unknown game preset '{game}' in {path} (available: {sorted(presets)})")
    cfg = dict(base) if base is not None else copy.deepcopy(DEFAULTS)
    for k, v in presets[game].items():
        if k not in DEFAULTS:
            raise ConfigError(f"unknown preset key '{k}' for game '{game}'")
        cfg[k] = v
    return validate_config(cfg)


def save_config(cfg: Mapping[str, Any], path: os.PathLike) -> None:
    from .serialization import write_text_atomic
    body = yaml.safe_dump({k: v for k, v in dict(cfg).items()}, sort_keys=True, default_flow_style=False)
    write_text_atomic(path, body)


def load_default_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULTS)
