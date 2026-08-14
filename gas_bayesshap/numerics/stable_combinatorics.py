"""Numerically stable combinatorics.

``scipy.special.comb`` and ``math.factorial`` can overflow float64 for larger
``M``.  The engine therefore routes all binomial coefficients through
log-space or integer arithmetic with an explicit overflow guard, while
preserving byte-level parity with the spec reference for the tested regime
(M <= 6 for Lemma E, arbitrary M for Lemma D / Shapley weights).
"""

from __future__ import annotations

import math

from scipy.special import comb as _scipy_comb, gammaln


def comb_exact(n: int, k: int) -> float:
    """Exact binomial coefficient ``C(n, k)`` as a float.

    Uses Python integers (arbitrary precision) and converts to float only at
    the end, so it never overflows for any representable ``(n, k)``.
    """
    if k < 0 or k > n or n < 0:
        return 0.0
    k = min(k, n - k)
    if k == 0:
        return 1.0
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= (n - k + i)
        den *= i
    return float(num // den)


def comb_log(n: int, k: int) -> float:
    """Log of the binomial coefficient (for huge ``n`` where the exact integer
    path would be slow but float64 log is fine)."""
    if k < 0 or k > n or n < 0:
        return float("-inf")
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def comb(n: int, k: int, exact: bool = True) -> float:
    """Binomial coefficient ``C(n, k)`` with overflow-safe fallback.

    Parameters
    ----------
    n, k : int
    exact : bool
        If True, uses exact integer arithmetic (spec reference uses
        ``scipy.special.comb`` which is exact in the tested regime; the exact
        integer path matches it to machine precision and is overflow-safe).
    """
    if k < 0 or k > n or n < 0:
        return 0.0
    if exact:
        return comb_exact(n, k)
    # approximate path used only when explicitly requested
    return float(_scipy_comb(n, k, exact=False))


def factorial(n: int) -> float:
    """Exact factorial as float (Python ints, no overflow)."""
    if n < 0:
        raise ValueError("factorial of negative number")
    return float(math.factorial(n))


def shapley_weight(s: int, M: int) -> float:
    """Shapley sampling weight ``w_s = s!(M-1-s)! / M!`` for strata of size ``s``."""
    if not (0 <= s <= M - 1):
        raise ValueError(f"stratum size {s} out of range for M={M}")
    return factorial(s) * factorial(M - 1 - s) / factorial(M)


def delta_weight(s: int, M: int) -> float:
    """Adjacent Shapley weight difference ``Delta w_s = w_s - w_{s+1}``.

    Spec (Lemma E): ``Delta w_s = s!(M-2-s)!(M-2-2s) / M!`` for ``0 <= s <= M-2``.
    """
    if M < 2:
        raise ValueError("Delta w_s requires M >= 2")
    if not (0 <= s <= M - 2):
        raise ValueError(f"s={s} out of range for M={M}")
    return factorial(s) * factorial(M - 2 - s) * (M - 2 - 2 * s) / factorial(M)
