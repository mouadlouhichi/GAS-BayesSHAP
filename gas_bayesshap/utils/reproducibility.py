"""Reproducibility: environment manifests and RNG state handling.

A repeated run with identical recorded state must reproduce results within
declared numerical tolerance.  This module captures everything the engine
needs to prove it (git commit, package versions, OS, CPU, RNG states).
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


def git_commit_and_dirty(repo_root: Optional[str] = None) -> Dict[str, str]:
    """Return ``{'commit': ..., 'dirty': ...}`` for the repository ('' if n/a)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        commit = out.stdout.strip() if out.returncode == 0 else ""
        out2 = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        dirty = bool(out2.stdout.strip()) if out2.returncode == 0 else True
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "", "dirty": True}


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ("numpy", "scipy", "PyYAML", "scikit-learn", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def cpu_info() -> str:
    try:
        import multiprocessing
        return f"{platform.processor()} x{multiprocessing.cpu_count()}"
    except Exception:
        return platform.processor() or "unknown"


def environment_manifest(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Full environment provenance record (spec section 38)."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": sys.version,
        "python_impl": platform.python_implementation(),
        "cpu": cpu_info(),
        "git": git_commit_and_dirty(repo_root),
        "packages": package_versions(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }


def save_rng_state(rng: np.random.RandomState) -> Dict[str, Any]:
    """Capture the full legacy RandomState (MT19937) state."""
    return {"random_state": rng.get_state()}


def restore_rng_state(rng: np.random.RandomState, state: Dict[str, Any]) -> np.random.RandomState:
    if state is None or "random_state" not in state:
        return rng
    rng.set_state(state["random_state"])
    return rng


def save_generator_state(rng: np.random.Generator) -> Dict[str, Any]:
    return {"generator_state": rng.bit_generator.state}


def restore_generator_state(rng: np.random.Generator, state: Dict[str, Any]) -> np.random.Generator:
    if state is not None and "generator_state" in state:
        rng.bit_generator.state = state["generator_state"]
    return rng


def make_rng(seed: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(0 if seed is None else int(seed))
