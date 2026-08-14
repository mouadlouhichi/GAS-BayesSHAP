"""Coupled adjacent-stratum Neyman allocation (Theorem A, spec sections 2.6/16).

In the add-one/remove-one scheme, drawing a coalition of cardinality ``q``
yields add-one samples in stratum ``q`` for ``(M - q)`` features and
remove-one samples in stratum ``q - 1`` for ``q`` features, so the expected
sample count backing interior stratum ``s`` is

.. math::

    n_s(\\mathbf{K}) = \\frac{M - s}{M} K_s + \\frac{s + 1}{M} K_{s + 1}.

The optimal draw distribution :math:`\\mathbf{K}^*` solves the **coupled**
convex program (never independently per-stratum):

.. math::

    \\min_{\\mathbf{K}} \\; \\frac{1}{M} \\sum_{s=1}^{M-2}
    \\frac{\\|\\boldsymbol{\\sigma}_s^r\\|_2^2}
        {(M-s) K_s + (s+1) K_{s+1}}
    \\quad \\text{s.t.} \\quad \\sum_{q=1}^{M-1} K_q = K_{\\text{cert}}, \\;
    K_q \\ge 0

Draw size ``q = M-1`` is explicitly included (it supplies remove-one samples
to the final interior stratum ``s = M-2``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class NeymanSolution:
    probabilities: np.ndarray      # (M,) — probs[0] == 0 always
    counts: np.ndarray             # (M,) integer counts for a given K_cert
    objective_value: float
    status: int                    # scipy status
    success: bool
    message: str
    K_cert: float
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {
            "probabilities": self.probabilities.tolist(),
            "counts": self.counts.tolist(),
            "objective_value": self.objective_value,
            "status": self.status,
            "success": self.success,
            "message": self.message,
            "K_cert": self.K_cert,
            "fallback_used": self.fallback_used,
        }


def _objective_from_probs(probs: np.ndarray, A: np.ndarray, M: int) -> float:
    K = np.zeros(M)
    K[1:M] = probs
    val = 0.0
    for s in range(1, M - 1):
        d_s = (M - s) * K[s] + (s + 1) * K[s + 1]
        val += A[s] / max(d_s, 1e-12)
    return val / M


def solve_coupled_neyman_allocation(
    sigma_res: np.ndarray,
    M: int,
    K_cert: float = 1.0,
    x0: Optional[np.ndarray] = None,
    maxiter: int = 1000,
) -> NeymanSolution:
    """Solve the coupled adjacent-stratum allocation program (SLSQP).

    The decision vector is ``K_1..K_{M-1}`` normalized to sum 1; returned
    probabilities have ``probs[0] = 0`` (extreme strata are never drawn in
    Stage 2).  Falls back to uniform over ``q in {1..M-1}`` on solver
    failure — **explicitly flagged** in the result.
    """
    M = int(M)
    sigma_res = np.asarray(sigma_res, dtype=np.float64)
    if M <= 2:
        return NeymanSolution(
            probabilities=np.zeros(M),
            counts=np.zeros(M, dtype=np.int64),
            objective_value=0.0,
            status=0,
            success=True,
            message="M <= 2: no interior strata, allocation is zero",
            K_cert=float(K_cert),
        )

    A = np.zeros(M)
    for s in range(1, M - 1):  # interior strata s = 1 .. M-2
        A[s] = np.sum(sigma_res[s, :] ** 2) + 1e-8

    def objective(K_dec):
        K = np.zeros(M)
        K[1:M] = K_dec
        val = 0.0
        for s in range(1, M - 1):
            d_s = (M - s) * K[s] + (s + 1) * K[s + 1]
            val += A[s] / max(d_s, 1e-12)
        return val / M

    n_q = M - 1
    if x0 is None:
        x0 = np.full(n_q, 1.0 / n_q)
    else:
        x0 = np.asarray(x0, dtype=np.float64).copy()
        if x0.shape[0] != n_q:
            raise ValueError(f"x0 must have length M-1={n_q}")
    bnds = [(0.0, 1.0) for _ in range(n_q)]
    cons = ({"type": "eq", "fun": lambda k: np.sum(k) - 1.0})

    # NOTE: default solver options — the spec reference calls SLSQP with
    # defaults; custom ftol/maxiter change the (flat) optimum and break
    # parity.  maxiter is only applied when explicitly requested.
    if maxiter != 1000:
        res = minimize(objective, x0, bounds=bnds, constraints=cons, method="SLSQP",
                       options={"maxiter": maxiter})
    else:
        res = minimize(objective, x0, bounds=bnds, constraints=cons, method="SLSQP")
    probs = np.zeros(M)
    fallback = False
    if res.success and np.sum(res.x) > 0:
        probs[1:M] = np.maximum(res.x, 0.0)
        probs[1:M] /= np.sum(probs[1:M])
    else:
        probs[1:M] = 1.0 / n_q
        fallback = True

    counts = np.zeros(M, dtype=np.int64)
    if K_cert > 0:
        counts = np.floor(K_cert * probs).astype(np.int64)
        # ensure the sum of counts respects K_cert as closely as possible
        diff = int(round(K_cert)) - int(counts.sum())
        if diff > 0:
            order = np.argsort(-(K_cert * probs - counts))
            for q in order:
                if diff <= 0:
                    break
                if q > 0:
                    counts[q] += 1
                    diff -= 1

    obj = objective(probs[1:M])
    return NeymanSolution(
        probabilities=probs,
        counts=counts,
        objective_value=float(obj),
        status=int(res.status),
        success=bool(res.success and not fallback),
        message=str(res.message),
        K_cert=float(K_cert),
        fallback_used=fallback,
    )
