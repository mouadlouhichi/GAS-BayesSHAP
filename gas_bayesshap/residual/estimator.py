"""Unbiased stratified residual Shapley estimator (spec section 26).

.. math::

    \\hat\\phi_i(r_\\mathcal{D}) = \\frac{1}{M} \\sum_{s=0}^{M-1}
        \\hat\\mu_{i,s}(R)

where :math:`\\hat\\mu_{i,s}` is the empirical mean of cell ``(i, s)``.
Empty cells are **not** silently skipped in the certification logic; the
estimator uses the deterministic extreme-stratum values (exact means) and
interior cells are guaranteed populated by the pilot before the adaptive
loop may certify (STRICT mode).
"""

from __future__ import annotations

import numpy as np

from .strata import StratumStore


def residual_shapley(store: StratumStore, M: int) -> np.ndarray:
    """Per-feature stratified residual estimate ``phi_hat(r_D)``."""
    phi_r = np.zeros(M, dtype=np.float64)
    for i in range(M):
        stratum_sum = 0.0
        for s in range(M):
            if store.count(i, s) > 0:
                stratum_sum += store.mean(i, s)
        phi_r[i] = stratum_sum / M
    return phi_r


def raw_unified_estimator(phi_m_b: np.ndarray, phi_hat_r: np.ndarray) -> np.ndarray:
    """``phi_hat^raw = phi(m_b) + phi_hat(r_D)`` (both components kept)."""
    return np.asarray(phi_m_b, dtype=np.float64) + np.asarray(phi_hat_r, dtype=np.float64)


def stratum_completeness(store: StratumStore, M: int):
    """Per-feature stratum-coverage completeness.

    Returns
    -------
    complete : np.ndarray of bool
        ``complete[i]`` is True iff every cell ``(s, i)`` for
        ``s = 0..M-1`` has at least one observation.
    missing : dict
        ``{i: [s, ...]}`` — the empty strata per incomplete feature.

    A feature with missing strata has a *partial* point estimate (its
    ``phi_res`` implicitly uses 0 for unobserved cells) and must never be
    presented as fully certified.
    """
    complete = np.ones(M, dtype=bool)
    missing: dict = {}
    for i in range(M):
        miss = [s for s in range(M) if store.count(i, s) == 0]
        if miss:
            complete[i] = False
            missing[i] = miss
    return complete, missing
