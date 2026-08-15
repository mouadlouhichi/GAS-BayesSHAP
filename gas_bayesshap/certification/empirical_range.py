"""Empirical residual-range tightening (Q1-review task #2, opt-in).

The spec's rigorous range ``R_delta_res = 4(U - L)`` is very conservative:
for a good GP surrogate the residual ``|v(S) - m_b(S)|`` is far smaller than
``(U - L)``, and the Bernstein range term ``7 R log / (3(n-1))`` dominates the
width.  This module provides **opt-in** tighter range modes:

``spec`` (default)
    ``R_eff = 4(U - L)`` — the rigorous, unchanged guarantee.

``empirical_max``
    ``R_eff = factor * max_i |observed residual|`` with a small-sample safety
    factor.  **Approximate**: the observed max underestimates the true
    support, so the anytime (1-delta) guarantee is NOT formally preserved.
    Must be flagged ``range_bound_is_heuristic = True`` in the result.

``holdout``
    ``R_eff = 2 * u_holdout`` where ``u_holdout`` is an upper bound on the
    surrogate error ``|v - m_b|`` estimated from a held-out set of coalition
    evaluations with the empirical-Bernstein inequality (distribution-free
    over the held-out coalitions; still heuristic over unseen coalitions).
    Flagged heuristic as well.

The purpose is (a) to quantify how much tighter the widths become in
practice (the review's key question), and (b) to demonstrate that with a
per-stratum empirical range, sign-certification becomes feasible at
K ~ 1e4-1e5 instead of 1e5-1e6.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def empirical_max_range(
    residuals: np.ndarray,
    safety_factor: float = 2.0,
    floor: float = 1e-6,
) -> float:
    """R_eff = safety_factor * max|residual| (approximate, heuristic).

    Parameters
    ----------
    residuals : observed residual marginals (per stratum or global).
    safety_factor : inflation of the observed max (>= 1).  Higher is safer;
                    2.0 is a common conservative heuristic.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.size == 0:
        return floor
    m = float(np.max(np.abs(residuals)))
    if not np.isfinite(m) or m <= 0:
        return floor
    return max(safety_factor * m, floor)


def holdout_surrogate_error_bound(
    v_holdout: np.ndarray,
    m_holdout: np.ndarray,
    delta: float = 0.05,
) -> float:
    """Upper bound on E|v - m_b| (or max) from a held-out set.

    Uses the empirical-Bernstein (Maurer-Pontil style) bound on the *mean*
    absolute surrogate error plus the observed max, so R_eff = 2 * bound is a
    conservative residual-marginal range on the observed coalitions.
    Distribution-free over the held-out coalitions; heuristic over unseen
    coalitions (flagged in the result).
    """
    e = np.abs(np.asarray(v_holdout, dtype=np.float64) - np.asarray(m_holdout, dtype=np.float64))
    n = e.size
    if n == 0:
        return 1.0
    mean = float(e.mean())
    var = float(e.var())
    if n < 2:
        return float(e.max()) * 2.0 + 1.0
    # empirical-Bernstein tail: P(mean - E[mean] > t) <= exp(-n t^2 / (2 var + (2/3) Rmax t))
    Rmax = float(e.max())
    t = math.sqrt((2 * var * math.log(2.0 / delta)) / n) + (2.0 * Rmax * math.log(2.0 / delta)) / (3.0 * n)
    bound = min(mean + t, Rmax)  # E|err| <= mean + t (and <= Rmax trivially)
    return float(bound)


def per_stratum_ranges(
    store,
    M: int,
    mode: str = "empirical_max",
    safety_factor: float = 2.0,
    delta: float = 0.05,
    spec_range: float = 4.0,
) -> np.ndarray:
    """Per-stratum effective residual range R_eff[s] (s = 0..M-1).

    ``mode="spec"`` -> every stratum uses ``spec_range`` (rigorous).
    ``mode="empirical_max"`` -> stratum range from observed |residual| max.
    ``mode="holdout"`` -> falls back to empirical_max per stratum (the true
    holdout variant is exposed as :func:`holdout_surrogate_error_bound`).
    """
    R_eff = np.full(M, float(spec_range))
    if mode == "spec":
        return R_eff
    for s in range(1, M - 1):  # extreme strata: exact, 0 width anyway
        vals = np.concatenate([store.values(i, s) for i in range(M)])
        if vals.size >= 1:
            R_eff[s] = empirical_max_range(vals, safety_factor=safety_factor)
    return R_eff


def report_width_improvement(spec_width: float, emp_width: float) -> float:
    """Ratio spec/empirical width (how many x tighter)."""
    return float(spec_width) / max(emp_width, 1e-12)
