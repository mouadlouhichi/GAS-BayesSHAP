"""Lemma D/E closed forms; brute force lives separately in game/reference.py."""
import numpy as np
from ..numerics.stable_combinatorics import choose, shapley_weight

def cross_covariance(mask, kernel):
    """Lemma D, O(M²) hypergeometric cross covariance A_i k(., mask)."""
    mask=np.asarray(mask,dtype=bool); M=mask.size; r=int(mask.sum()); rho=kernel.rho; scale=kernel.sigma0**2
    def term(r_without_i, sign):
        total=0.
        for s in range(M):
            den=choose(M-1,s)
            for l in range(max(0,s-(M-1-r_without_i)),min(s,r_without_i)+1):
                total += choose(r_without_i,l)*choose(M-1-r_without_i,s-l)/den*rho**(r_without_i+s-2*l)
        return sign*scale*(1-rho)*total/M
    vin=term(r-1,1.) if r else 0.; vout=term(r,-1.) if r<M else 0.
    return np.where(mask,vin,vout)

def prior_shapley_covariance(M,kernel):
    """Lemma E with raw pair counts and mandatory Delta-w factors."""
    rho=kernel.rho; scale=kernel.sigma0**2
    diag_sum=0.
    for s in range(M):
      for t in range(M):
       den=choose(M-1,t)
       for l in range(max(0,s+t-M+1),min(s,t)+1):
        diag_sum += choose(s,l)*choose(M-1-s,t-l)/den*rho**(s+t-2*l)
    diag=2*scale*(1-rho)*diag_sum/(M*M)
    off=0.
    for s in range(max(0,M-1)):
      ds=shapley_weight(M,s)-shapley_weight(M,s+1)
      for t in range(M-1):
       dt=shapley_weight(M,t)-shapley_weight(M,t+1)
       for l in range(max(0,s+t-M+2),min(s,t)+1):
        off += ds*dt*choose(M-2,s)*choose(s,l)*choose(M-2-s,t-l)*rho**(s+t-2*l)
    off *= scale*(1-rho)**2
    return (diag-off)*np.eye(M)+off*np.ones((M,M))

def posterior_shapley_covariance(prior,kphiD,inv):
    return (prior-kphiD@inv@kphiD.T + (prior-kphiD@inv@kphiD.T).T)/2
