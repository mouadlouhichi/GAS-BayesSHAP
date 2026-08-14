import numpy as np
import pytest
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel
from gas_bayesshap.gp.covariance import cross_covariance,prior_shapley_covariance
from gas_bayesshap.game.reference import cross_covariance_bruteforce,covariance_bruteforce
@pytest.mark.parametrize('m',range(1,7))
def test_lemma_d(m):
 k=ExponentialHammingKernel();s=np.arange(m)%2==0
 assert np.allclose(cross_covariance(s,k),cross_covariance_bruteforce(s,k))
@pytest.mark.parametrize('m',range(2,7))
def test_lemma_e(m):
 k=ExponentialHammingKernel();assert np.allclose(prior_shapley_covariance(m,k),covariance_bruteforce(m,k))
