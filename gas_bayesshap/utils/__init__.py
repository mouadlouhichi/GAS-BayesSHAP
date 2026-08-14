"""Utility layer: config, hashing, reproducibility, serialization, RNG."""

from .config import ConfigError, load_config, load_default_config, validate_config
from .hashing import (
    background_hash,
    cache_key,
    config_hash,
    input_hash,
    oracle_hash,
    stable_hash,
)
from .reproducibility import environment_manifest, git_commit_and_dirty, make_rng
from .rng_state import dict_to_rng_state, rng_state_hash, rng_state_to_dict
from .serialization import (
    ensure_dir,
    jsonable,
    load_json,
    load_npz,
    write_json_atomic,
    write_npz_atomic,
    write_text_atomic,
)

__all__ = [
    "ConfigError",
    "load_config",
    "load_default_config",
    "validate_config",
    "background_hash",
    "cache_key",
    "config_hash",
    "input_hash",
    "oracle_hash",
    "stable_hash",
    "environment_manifest",
    "git_commit_and_dirty",
    "make_rng",
    "dict_to_rng_state",
    "rng_state_hash",
    "rng_state_to_dict",
    "ensure_dir",
    "jsonable",
    "load_json",
    "load_npz",
    "write_json_atomic",
    "write_npz_atomic",
    "write_text_atomic",
]
