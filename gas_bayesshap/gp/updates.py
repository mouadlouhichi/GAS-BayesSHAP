"""Sherman-Morrison rank-1 inverse updates (spec section 20).

Incremental inversion of the Gram matrix ``K_DD + eta^2 I`` when a new
observation is appended:

.. math::

    \\begin{pmatrix}
        A^{-1} + s^{-1} vv^T & -s^{-1} v \\\\
        -s^{-1} v^T         & s^{-1}
    \\end{pmatrix},
    \\qquad s = k_{self} + \\eta^2 - k^T A^{-1} k

with the **near-duplicate guard**: if the Schur complement :math:`s < \\eta^2`
the update is rejected (``ok=False``) to prevent numerical blowup, exactly as
in the spec reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class UpdateResult:
    inv_K: np.ndarray
    ok: bool
    schur: float
    threshold: float
    action: str  # "accepted" | "rejected_near_duplicate"
    k_self: float


def rank1_inverse_update(
    inv_K: np.ndarray,
    k_vec: np.ndarray,
    k_self: float,
    eta_sq: float,
) -> Tuple[np.ndarray, bool]:
    """Sherman-Morrison block update in O(D^2) time.

    Returns ``(new_inv, ok)`` where ``ok=False`` means the candidate was a
    near-duplicate (``schur < eta^2``) and ``inv_K`` is returned unchanged.
    """
    D = inv_K.shape[0]
    if D == 0:
        return np.array([[1.0 / (k_self + eta_sq)]], dtype=np.float64), True

    v = inv_K @ k_vec
    schur = (k_self + eta_sq) - float(k_vec.T @ v)
    if schur < eta_sq:
        return inv_K, False  # near duplicate, skip update

    schur_inv = 1.0 / max(schur, 1e-10)
    top_left = inv_K + (schur_inv * np.outer(v, v))
    top_right = (-schur_inv * v).reshape(-1, 1)
    bottom_left = (-schur_inv * v).reshape(1, -1)
    bottom_right = np.array([[schur_inv]], dtype=np.float64)
    return np.block([[top_left, top_right], [bottom_left, bottom_right]]), True


def rank1_inverse_update_detailed(
    inv_K: np.ndarray,
    k_vec: np.ndarray,
    k_self: float,
    eta_sq: float,
) -> UpdateResult:
    """Detailed variant of :func:`rank1_inverse_update` with diagnostics."""
    D = inv_K.shape[0]
    if D == 0:
        return UpdateResult(
            inv_K=np.array([[1.0 / (k_self + eta_sq)]], dtype=np.float64),
            ok=True,
            schur=k_self + eta_sq,
            threshold=eta_sq,
            action="accepted",
            k_self=k_self,
        )
    v = inv_K @ k_vec
    schur = (k_self + eta_sq) - float(k_vec.T @ v)
    if schur < eta_sq:
        return UpdateResult(inv_K, False, schur, eta_sq, "rejected_near_duplicate", k_self)
    schur_inv = 1.0 / max(schur, 1e-10)
    top_left = inv_K + (schur_inv * np.outer(v, v))
    top_right = (-schur_inv * v).reshape(-1, 1)
    bottom_left = (-schur_inv * v).reshape(1, -1)
    bottom_right = np.array([[schur_inv]], dtype=np.float64)
    new_inv = np.block([[top_left, top_right], [bottom_left, bottom_right]])
    return UpdateResult(new_inv, True, schur, eta_sq, "accepted", k_self)
