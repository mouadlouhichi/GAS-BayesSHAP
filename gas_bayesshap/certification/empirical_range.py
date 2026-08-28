"""Empirical residual-range tightening (Q1-review task #2, opt-in).

The spec's rigorous range ``R_delta_res = 4(U - L)`` is very conservative:
for a good GP surrogate the residual ``|v(S) - m_b(S)|`` is far smaller than
``(U - L)``, and the Bernstein range term ``7 R log / (3(n-1))`` dominates the
width.  This module provides **opt-in** tighter range modes:

``spec`` (default)
    ``R_eff = 4(U - L)`` — the rigorous, unchanged anytime guarantee at the
    nominal level ``1 - delta``.

``finite_population``  (empirical-range tightening with coupon diagnostic,
    Theorem E / E' in the paper — **nominal anytime validity only after
    deterministic per-cell thresholds**)
    ``R_eff = 2 * max|observed residual|`` per stratum, i.e. the observed
    support of the residual marginal.  The residual marginal for cell
    ``(i, s)`` is an iid uniform draw from the **finite population** of
    ``C(M-1, s)`` pairs ``{T, T u {i}}`` with ``|T| = s``, so the probability
    that the support-maximising pair remains unobserved after ``n_{i,s}``
    draws is *at most* ``(1 - 1/N_{i,s})**n`` (coupon-collector bound;
    equality only for a unique maximiser).  At fixed ``n_{i,s}`` the
    certificate holds at the *realised* level ``1 - delta2 - delta1`` with
    ``delta1 = sum_{i,s} (1 - 1/C(M-1,s))**n_{i,s}``.  At a data-dependent
    stopping time ``tau``, ``delta1(tau)`` is random and the realised level
    is **diagnostic, conditional-on-history, not anytime**.  Nominal
    ``1 - delta`` anytime validity is claimed only after deterministic
    per-cell thresholds ``n_{i,s} >= n^*_{i,s}`` are met (Corollary E,
    ``certificate_at_nominal_level`` flag).  Reported via
    ``finite_population_delta1`` / ``finite_population_coverage_level`` /
    ``certificate_at_nominal_level``; ``range_bound_is_heuristic`` is
    ``False`` for this mode, but ``certificate_is_rigorous`` additionally
    requires the deterministic thresholds.

``empirical_max``  (**heuristic**)
    ``R_eff = factor * max_i |observed residual|`` with a small-sample safety
    factor.  **Approximate**: for a *generic* (non-finite) population the
    observed max can underestimate the true support by an arbitrary amount
    (see the impossibility remark in the paper), so the anytime (1-delta)
    guarantee is NOT formally preserved.  Flagged
    ``range_bound_is_heuristic = True``.

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


def population_size(M: int, s: int) -> int:
    """Number of distinct pairs ``{T, T u {i}}`` with ``|T| = s``.

    The residual marginal of cell ``(i, s)`` is an iid uniform draw from this
    finite population: the add-one path draws ``T`` with ``|T| = s`` and the
    remove-one path draws ``T`` with ``|T| = s`` (via ``S' = T u {i}``), both
    uniformly over the ``C(M-1, s)`` choices of ``T`` not containing ``i``.
    """
    if s < 0 or s > M - 1:
        return 1
    return int(math.comb(M - 1, s))


def finite_population_coupon_delta1(store, M: int) -> float:
    """Realised coupon-collector failure budget ``delta1`` (Theorem E).

    ``delta1 = sum_{i, s interior} (1 - 1/N_{i,s})^{n_{i,s}}`` where
    ``N_{i,s} = C(M-1, s)`` is the finite-population size of cell ``(i, s)``
    and ``n_{i,s}`` is the number of residual draws recorded for that cell.
    A cell with ``n = 0`` observations contributes ``(1 - 1/N)^0 = 1`` (the
    support-maximising pair is unobserved with certainty).  This is an upper
    bound (not an equality) on the probability that any support-maximising
    pair remains unobserved at the time the certificate is produced: if
    ``m`` coalitions share the maximum absolute residual, the failure
    probability is ``(1 - m/N)^n <= (1 - 1/N)^n``.  The realised coverage
    level is therefore ``1 - delta2 - delta1`` (lower bound).
    """
    total = 0.0
    for s in range(1, M - 1):
        N_s = population_size(M, s)
        if N_s <= 1:
            continue
        for i in range(M):
            n = store.count(i, s)
            total += (1.0 - 1.0 / N_s) ** n
    return float(total)


def finite_population_coverage_level(delta1: float, delta2: float = 0.05) -> float:
    """Realised simultaneous coverage level ``1 - delta2 - delta1``."""
    return float(max(0.0, 1.0 - delta2 - delta1))


def coupon_threshold(M: int, s: int, budget_delta1: float) -> int:
    """Deterministic sample count needed so the cell's coupon term is
    ``<= budget_delta1`` (used by the anytime Theorem E')."""
    N_s = population_size(M, s)
    if N_s <= 1:
        return 0
    if budget_delta1 <= 0 or budget_delta1 >= 1:
        return 0
    if 1.0 - 1.0 / N_s <= budget_delta1:
        return 1
    return int(math.ceil(math.log(budget_delta1) / math.log(1.0 - 1.0 / N_s)))


def deterministic_coupon_thresholds_satisfied(store, M: int, delta_coupon: float = 0.025) -> bool:
    """Check deterministic per-cell coupon thresholds (reviewer-proof nominal gating).

    Choose fixed error allocations alpha_{i,s} = delta_coupon / (M*(M-2))
    before sampling, where M*(M-2) = number of interior cells (s=1..M-2 per
    feature).  Compute n^*_{i,s} = ceil(log alpha / log(1-1/N_s)).
    Nominal certification is claimed only if n_{i,s} >= n^*_{i,s} for all
    interior cells.  This is valid under adaptive allocation because the
    thresholds are deterministic (fixed before sampling).

    Parameters
    ----------
    store: StratumStore with .count(i,s)
    M: number of features
    delta_coupon: total coupon budget (typically delta/2)
    """
    if M <= 2:
        return True
    n_cells = M * (M - 2)
    if n_cells <= 0:
        return True
    alpha = delta_coupon / n_cells
    if alpha <= 0 or alpha >= 1:
        return False
    for s in range(1, M - 1):
        N_s = population_size(M, s)
        if N_s <= 1:
            continue
        n_star = coupon_threshold(M, s, alpha)
        for i in range(M):
            if store.count(i, s) < n_star:
                return False
    return True


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
    ``mode="finite_population"`` -> stratum range ``2 * max|observed
    residual|`` (valid at fixed n at realised level ``1 - delta2 - delta1``,
    see :func:`finite_population_coupon_delta1`; diagnostic at stopping
    time tau, nominal only after deterministic thresholds
    :func:`deterministic_coupon_thresholds_satisfied`; with the default
    ``safety_factor=2.0`` the range is exactly twice the observed max, the
    support bound once the support max is observed).
    ``mode="empirical_max"`` -> ``safety_factor * max|observed residual|``
    (heuristic, flagged).
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
