"""Overflow-safety and correctness of the combinatorics layer."""

import math

import numpy as np

from gas_bayesshap.numerics.stable_combinatorics import comb, comb_exact, delta_weight


def test_no_float_overflow():
    # scipy.special.comb would return inf for some larger n; our exact path
    # must remain finite and correct.
    for n, k in [(100, 50), (150, 75), (200, 100)]:
        val = comb(n, k)
        assert math.isfinite(val)
        assert val > 0


def test_consistent_with_scipy_small():
    from scipy.special import comb as sp_comb
    for n in range(0, 10):
        for k in range(0, n + 1):
            assert comb(n, k) == float(sp_comb(n, k, exact=True))
            assert comb_exact(n, k) == float(sp_comb(n, k, exact=True))


def test_delta_weights_finite_large_M():
    M = 50
    dw = np.array([delta_weight(s, M) for s in range(M - 1)])
    assert np.all(np.isfinite(dw))
