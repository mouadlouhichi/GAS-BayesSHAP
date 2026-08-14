"""Tier-1: Lemma D analytical cross-covariance == brute-force enumeration."""

import numpy as np
import pytest

from gas_bayesshap.game.brute_force import brute_force_cross_covariance
from gas_bayesshap.kernels.covariance import lemma_D_cross_cov
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel

SIGMA0 = 1.0
LENGTHSCALE = 1.5
ATOL = 1e-10


def make_kernel():
    return ExponentialHammingKernel(sigma0=SIGMA0, lengthscale=LENGTHSCALE)


def test_lemma_D_m1_sanity():
    """Lemma D M=1 sanity check (spec section 44)."""
    k = make_kernel()
    for S_j in (np.array([False]), np.array([True])):
        analytic = lemma_D_cross_cov(k, S_j, 1)
        brute = brute_force_cross_covariance(k, S_j, 1)
        assert np.allclose(analytic, brute, atol=ATOL)


def test_lemma_D_m4_sign_and_exact_enumeration():
    """Spec test 1: M=4, sign pattern and machine-precision match."""
    M = 4
    k = make_kernel()
    S_j = np.array([True, False, True, False])
    analytic = lemma_D_cross_cov(k, S_j, M)
    brute = brute_force_cross_covariance(k, S_j, M)
    assert np.allclose(analytic, brute, atol=ATOL)

    # sign structure: i in S_j gets +, i notin S_j gets -
    r = int(np.sum(S_j))
    assert analytic[0] > 0 and analytic[2] > 0
    assert analytic[1] < 0 and analytic[3] < 0

    # symmetry: all in-members share V_in, all out-members share V_out
    assert abs(analytic[0] - analytic[2]) < ATOL
    assert abs(analytic[1] - analytic[3]) < ATOL


@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lemma_D_multiple_coalitions(M, seed):
    k = make_kernel()
    rng = np.random.RandomState(seed)
    for _ in range(6):
        p = np.zeros(M, dtype=bool)
        p[rng.permutation(M)[: rng.randint(0, M + 1)]] = True
        a = lemma_D_cross_cov(k, p, M)
        b = brute_force_cross_covariance(k, p, M)
        assert np.max(np.abs(a - b)) <= ATOL


def test_lemma_D_empty_and_full():
    k = make_kernel()
    M = 4
    empty = np.zeros(M, dtype=bool)
    full = np.ones(M, dtype=bool)
    assert np.allclose(lemma_D_cross_cov(k, empty, M),
                       brute_force_cross_covariance(k, empty, M), atol=ATOL)
    assert np.allclose(lemma_D_cross_cov(k, full, M),
                       brute_force_cross_covariance(k, full, M), atol=ATOL)
