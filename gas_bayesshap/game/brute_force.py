"""Independent brute-force reference engine (validation oracle only).

Every function here is a self-contained, direct enumeration that **never**
calls the optimized implementation.  It is used exclusively to validate the
optimized Lemmas D / E and to produce exact Shapley ground truth for
small ``M`` (mandatory: M in {2, 3, 4, 5, 6}).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from ..numerics.stable_combinatorics import factorial
from .subsets import all_subsets


def _w(s: int, M: int) -> float:
    return factorial(s) * factorial(M - 1 - s) / factorial(M)


def brute_force_shapley(game_fn: Callable[[np.ndarray], float], M: int) -> np.ndarray:
    """Exact Shapley values via full :math:`2^M` enumeration.

    .. math::

        \\phi_i(v) = \\sum_{S \\subseteq N\\setminus\\{i\\}}
            \\frac{|S|!(M-|S|-1)!}{M!}\\,[v(S\\cup\\{i\\}) - v(S)]

    ``game_fn`` must be the deterministic set-function value ``v(S)``.
    """
    M = int(M)
    values: dict[int, float] = {}

    def v(mask) -> float:
        if mask not in values:
            values[mask] = float(game_fn(mask_to_bool(mask, M)))
        return values[mask]

    phi = np.zeros(M, dtype=np.float64)
    for mask in range(1 << M):
        S = mask_to_bool(mask, M)
        s = int(np.sum(S))
        for i in range(M):
            if S[i]:
                continue
            Su = mask | (1 << i)
            phi[i] += _w(s, M) * (v(Su) - v(mask))
    return phi


def mask_to_bool(mask: int, M: int) -> np.ndarray:
    return np.array([(mask >> bit) & 1 for bit in range(M)], dtype=bool)


def brute_force_cross_covariance(kernel, S_j: np.ndarray, M: int) -> np.ndarray:
    """Lemma D by explicit Shapley-weighted enumeration over all :math:`2^M` subsets."""
    M = int(M)
    S_j = np.asarray(S_j, dtype=bool)
    K = np.zeros(M, dtype=np.float64)
    for i in range(M):
        for mask in range(1 << M):
            S = mask_to_bool(mask, M)
            if S[i]:
                continue
            s = int(np.sum(S))
            Su = S.copy()
            Su[i] = True
            d_u = int(np.sum(Su != S_j))
            d_s = int(np.sum(S != S_j))
            K[i] += _w(s, M) * (kernel.k(Su, S_j) - kernel.k(S, S_j))
    return K


def brute_force_prior_covariance(kernel, M: int) -> np.ndarray:
    """Lemma E by :math:`4^M` double enumeration of ``A_i A_j' k(S, T)``."""
    M = int(M)
    K = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            for m1 in range(1 << M):
                S = mask_to_bool(m1, M)
                if S[i]:
                    continue
                s = int(np.sum(S))
                ws = _w(s, M)
                for m2 in range(1 << M):
                    T = mask_to_bool(m2, M)
                    if T[j]:
                        continue
                    t = int(np.sum(T))
                    Su, Tu = S.copy(), T.copy()
                    Su[i] = True
                    Tu[j] = True
                    k11 = kernel.k(Su, Tu)
                    k10 = kernel.k(Su, T)
                    k01 = kernel.k(S, Tu)
                    k00 = kernel.k(S, T)
                    K[i, j] += ws * _w(t, M) * (k11 - k10 - k01 + k00)
    return K


def exact_game_values(
    oracle: "object",
    x: np.ndarray,
    M: int,
) -> dict:
    """Enumerate all :math:`2^M` coalition values through an oracle (with caching).

    Returns ``{bitmask: value}``.  Each value is obtained from the oracle, so
    query accounting remains honest (every coalition is evaluated once).
    """
    values: dict[int, float] = {}
    for mask in range(1 << M):
        S = mask_to_bool(mask, M)
        values[mask] = float(oracle.evaluate(x, S))
    return values


def exact_shapley_from_values(values: dict, M: int) -> np.ndarray:
    """Shapley values from a dict of ``{bitmask: v(S)}``."""
    phi = np.zeros(M, dtype=np.float64)
    for mask, val in values.items():
        S = mask_to_bool(mask, M)
        s = int(np.sum(S))
        for i in range(M):
            if S[i]:
                continue
            Su = mask | (1 << i)
            phi[i] += _w(s, M) * (values[Su] - val)
    return phi
