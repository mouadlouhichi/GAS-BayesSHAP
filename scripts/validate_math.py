import numpy as np
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel
from gas_bayesshap.gp.covariance import cross_covariance,prior_shapley_covariance
from gas_bayesshap.game.reference import cross_covariance_bruteforce,covariance_bruteforce
for m in range(1,7):
 k=ExponentialHammingKernel(1,1.5); s=np.array([(i%2)==0 for i in range(m)])
 assert np.allclose(cross_covariance(s,k),cross_covariance_bruteforce(s,k),atol=1e-10)
 if m>=2: assert np.allclose(prior_shapley_covariance(m,k),covariance_bruteforce(m,k),atol=1e-10)
print('Lemma D (M=1..6) and Lemma E (M=2..6): PASS')
