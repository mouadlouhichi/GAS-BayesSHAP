"""Numeric validation primitives used across the engine."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


class NumericalFailure(RuntimeError):
    """Raised when a numerical invariant is violated (NaN/Inf/singularity)."""


def assert_finite(arr, name: str = "array") -> None:
    """Raise NumericalFailure if ``arr`` contains NaN or Inf."""
    a = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        bad = np.sum(~np.isfinite(a))
        raise NumericalFailure(f"{name} contains {bad} non-finite entries")


def assert_shape(arr, shape: Iterable[int], name: str = "array") -> None:
    a = np.asarray(arr)
    if tuple(a.shape) != tuple(shape):
        raise NumericalFailure(f"{name} shape {a.shape} != expected {tuple(shape)}")


def require_finite(arr, name: str = "array", fallback=None):
    """Return finite ``arr`` or ``fallback``; raises if no fallback given."""
    a = np.asarray(arr, dtype=np.float64)
    if np.all(np.isfinite(a)):
        return a
    if fallback is not None:
        return fallback
    raise NumericalFailure(f"{name} contains non-finite values")


def safe_std(arr, default: float = 0.5) -> float:
    """Sample standard deviation (ddof=1) preserving true zero variance.

    Spec reference semantics:
    - len <= 1            -> ``default``
    - non-finite result   -> ``default``
    - otherwise           -> ``max(std, 0.0)``
    """
    if len(arr) <= 1:
        return default
    val = float(np.std(np.asarray(arr, dtype=np.float64), ddof=1))
    if not np.isfinite(val):
        return default
    return max(val, 0.0)


def assert_in_unit_range(values, name: str = "values", tol: float = 1e-10) -> None:
    v = np.asarray(values, dtype=np.float64)
    if np.any(v < -tol) or np.any(v > 1.0 + tol):
        raise NumericalFailure(f"{name} outside [0, 1] (min={v.min():.3e}, max={v.max():.3e})")
