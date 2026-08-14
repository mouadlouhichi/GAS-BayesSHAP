"""Numerical linear algebra helpers for the GP engine."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .validation import NumericalFailure, assert_finite


def sym(A: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix in-place-safe way: ``0.5 * (A + A^T)``."""
    return 0.5 * (A + A.T)


def solve_inverse(K: np.ndarray, eta_sq: float) -> np.ndarray:
    """Direct inverse of ``(K + eta^2 I)`` with finite check (reference path)."""
    n = K.shape[0]
    inv = np.linalg.inv(K + eta_sq * np.eye(n))
    assert_finite(inv, "Gram inverse")
    return inv


def solve_system(K: np.ndarray, y: np.ndarray, eta_sq: float) -> np.ndarray:
    """Solve ``(K + eta^2 I) alpha = y`` (numerically safer than explicit inverse)."""
    n = K.shape[0]
    alpha = np.linalg.solve(K + eta_sq * np.eye(n), y)
    assert_finite(alpha, "linear solve")
    return alpha


def posterior_covariance(
    K_phi_phi: np.ndarray,
    K_phi_D: np.ndarray,
    inv_K_DD: np.ndarray,
) -> np.ndarray:
    """Posterior Shapley covariance ``K_phi_phi - K_phi_D inv(K_DD) K_phi_D^T``.

    Symmetrized numerically (spec reference does ``0.5*(A + A.T)``).
    """
    phi_cov = K_phi_phi - (K_phi_D @ inv_K_DD @ K_phi_D.T)
    phi_cov = sym(phi_cov)
    assert_finite(phi_cov, "posterior covariance")
    return phi_cov


def posterior_variances(phi_cov: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Diagonal posterior variances with a positivity floor."""
    return np.maximum(np.diag(phi_cov), floor)


def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(A, A.T, atol=tol, rtol=0.0)


def check_symmetric(A: np.ndarray, name: str, tol: float = 1e-10) -> None:
    if A.shape[0] != A.shape[1]:
        raise NumericalFailure(f"{name} is not square: {A.shape}")
    if not is_symmetric(A, tol=tol):
        raise NumericalFailure(f"{name} is not symmetric (max |A-A^T| = {np.max(np.abs(A - A.T)):.3e})")


def is_psd(A: np.ndarray, tol: float = 1e-9) -> bool:
    eigvals = np.linalg.eigvalsh(A)
    return bool(np.min(eigvals) >= -tol * max(1.0, np.max(np.abs(eigvals))))


def safe_divide(numerators: np.ndarray, denominators: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return numerators / np.maximum(denominators, eps)
