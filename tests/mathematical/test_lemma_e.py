"""Tier-2: Lemma E analytical prior Shapley covariance == 4^M brute force."""

import numpy as np
import pytest

from gas_bayesshap.game.brute_force import brute_force_prior_covariance
from gas_bayesshap.kernels.covariance import lemma_E_prior_cov
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel

SIGMA0 = 1.0
LENGTHSCALE = 1.5
ATOL = 1e-10


@pytest.mark.parametrize("M", [2, 3, 4, 5, 6])
def test_lemma_E_matches_brute_force(M):
    k = ExponentialHammingKernel(sigma0=SIGMA0, lengthscale=LENGTHSCALE)
    analytic = lemma_E_prior_cov(k, M)
    brute = brute_force_prior_covariance(k, M)
    diff = np.max(np.abs(analytic - brute))
    assert diff < ATOL, f"Lemma E failed at M={M}: max|diff|={diff:.3e}"


def test_lemma_E_m2_off_diagonal():
    """Spec section 12: explicitly verify the M=2 off-diagonal covariance."""
    M = 2
    k = ExponentialHammingKernel(sigma0=SIGMA0, lengthscale=LENGTHSCALE)
    analytic = lemma_E_prior_cov(k, M)
    brute = brute_force_prior_covariance(k, M)
    assert abs(analytic[0, 1] - brute[0, 1]) < ATOL
    assert abs(analytic[1, 0] - brute[1, 0]) < ATOL
    # exchangeable structure: off-diagonal constant, diagonal constant
    assert abs(analytic[0, 1] - analytic[1, 0]) < ATOL
    assert abs(analytic[0, 0] - analytic[1, 1]) < ATOL


def test_lemma_E_structure():
    """(V_diag - V_off) I + V_off 1 1^T structure."""
    M = 5
    k = ExponentialHammingKernel(sigma0=SIGMA0, lengthscale=LENGTHSCALE)
    K = lemma_E_prior_cov(k, M)
    v_diag = K[0, 0]
    v_off = K[0, 1]
    expected = (v_diag - v_off) * np.eye(M) + v_off * np.ones((M, M))
    assert np.allclose(K, expected, atol=1e-12)
    # symmetric (post-symmetrization)
    assert np.allclose(K, K.T, atol=1e-12)


def test_lemma_E_delta_w_factor_present():
    """The Delta w_s factors must matter: without them the off-diagonal is wrong."""
    M = 6
    k = ExponentialHammingKernel(sigma0=SIGMA0, lengthscale=LENGTHSCALE)
    analytic = lemma_E_prior_cov(k, M)
    brute = brute_force_prior_covariance(k, M)
    # If someone drops the (M-2-2s) factor the result will not match brute force.
    assert np.allclose(analytic, brute, atol=ATOL)
