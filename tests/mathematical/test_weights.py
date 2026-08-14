"""Shapley weight sanity checks (w_s and Delta w_s)."""

import math

import numpy as np
import pytest

from gas_bayesshap.numerics.stable_combinatorics import (
    comb,
    comb_exact,
    delta_weight,
    factorial,
    shapley_weight,
)


def test_weights_sum_to_one():
    """The Shapley weights sum to 1 when averaged over subsets:
    sum_s C(M-1, s) * w_s = 1."""
    for M in range(1, 8):
        ws = np.array([shapley_weight(s, M) for s in range(M)])
        counts = np.array([comb(M - 1, s) for s in range(M)])
        assert abs(np.dot(counts, ws) - 1.0) < 1e-12


def test_weight_formula():
    M = 5
    for s in range(M):
        expected = math.factorial(s) * math.factorial(M - 1 - s) / math.factorial(M)
        assert abs(shapley_weight(s, M) - expected) < 1e-15


def test_delta_w_formula():
    M = 6
    for s in range(M - 1):
        expected = (
            math.factorial(s)
            * math.factorial(M - 2 - s)
            * (M - 2 - 2 * s)
            / math.factorial(M)
        )
        assert abs(delta_weight(s, M) - expected) < 1e-12
        # and it equals w_s - w_{s+1}
        assert abs(delta_weight(s, M) - (shapley_weight(s, M) - shapley_weight(s + 1, M))) < 1e-12


def test_comb_matches_scipy_small():
    from scipy.special import comb as sp_comb
    for n in range(0, 12):
        for k in range(0, n + 1):
            assert comb(n, k) == float(sp_comb(n, k, exact=True))
            assert comb_exact(n, k) == float(sp_comb(n, k, exact=True))


def test_comb_no_overflow_large():
    # float64 would overflow for C(200, 100); exact integer path must not
    assert comb(200, 100) > 0
    assert math.isfinite(comb(200, 100))


def test_comb_edge_cases():
    assert comb(0, 0) == 1.0
    assert comb(5, -1) == 0.0
    assert comb(5, 6) == 0.0
    assert comb(5, 5) == 1.0
    assert comb(5, 0) == 1.0


def test_factorial():
    assert factorial(0) == 1.0
    assert factorial(10) == math.factorial(10)
    with pytest.raises(ValueError):
        factorial(-1)
