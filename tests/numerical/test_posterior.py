"""GP posterior reference comparison (posterior covariance vs direct)."""

import numpy as np

from gas_bayesshap.gp.control_variate import BoundedLinearSurrogate, fit_bounded_surrogate
from gas_bayesshap.gp.posterior import gp_predict, gp_posterior, validate_surrogate_boundedness
from gas_bayesshap.kernels.covariance import lemma_D_cross_cov_matrix, lemma_E_prior_cov
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel


def make_design(M=5, D=7, seed=0):
    rng = np.random.RandomState(seed)
    coals = []
    for _ in range(D):
        p = np.zeros(M, dtype=bool)
        p[rng.permutation(M)[: rng.randint(1, M)]] = True
        coals.append(p)
    y = rng.randn(D)
    return coals, y


def test_predict_matches_manual():
    kernel = ExponentialHammingKernel(1.0, 1.5)
    coals, y = make_design()
    sur = fit_bounded_surrogate(coals, list(y), kernel, eta=1e-4)
    S = np.array([True, False, True, False, True])
    # manual h(S) = k_D(S)^T alpha
    d_H = np.sum(np.asarray(coals, dtype=bool) != S[None, :], axis=1)
    k_vec = kernel.sigma0_sq * (kernel.rho ** d_H)
    manual = float(k_vec @ sur.alpha)
    assert abs(sur.predict(S, kernel) - manual) < 1e-12


def test_gp_posterior_matches_direct():
    M = 5
    kernel = ExponentialHammingKernel(1.0, 1.5)
    coals, y = make_design(M=M, D=7)
    D = np.array(coals, dtype=bool)
    K_DD = kernel.gram(D)
    inv = np.linalg.inv(K_DD + 1e-8 * np.eye(len(coals)))
    K_phi_D = lemma_D_cross_cov_matrix(kernel, D, M)
    K_phi_phi = lemma_E_prior_cov(kernel, M)
    cov, vars_ = gp_posterior(K_phi_phi, K_phi_D, inv, scale=0.7)
    expected = (0.7 ** 2) * (K_phi_phi - K_phi_D @ inv @ K_phi_D.T)
    assert np.allclose(cov, expected, atol=1e-10)
    assert np.allclose(vars_, np.maximum(np.diag(expected), 1e-10), atol=1e-12)


def test_surrogate_boundedness_sweep():
    M = 4
    kernel = ExponentialHammingKernel(1.0, 1.5)
    rng = np.random.RandomState(1)
    coals = [np.zeros(M, dtype=bool), np.ones(M, dtype=bool)]
    for _ in range(10):
        p = np.zeros(M, dtype=bool)
        p[rng.permutation(M)[: rng.randint(0, M + 1)]] = True
        coals.append(p)
    y = [float(np.mean(c)) for c in coals]
    sur = fit_bounded_surrogate(coals, y, kernel, eta=1e-4)
    # apply bounds [0, 1]
    sur.scale = min(1.0, 1.0 / max(sur.h_ub - sur.h_lb, 1e-12))
    sur.shift = 0.0 - sur.scale * sur.h_lb
    assert validate_surrogate_boundedness(
        sur.D_coalitions, sur.alpha, kernel, sur.scale, sur.shift, M, 0.0, 1.0, tol=1e-9
    )
