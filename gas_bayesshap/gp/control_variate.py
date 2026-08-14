import numpy as np
from .covariance import cross_covariance, prior_shapley_covariance, posterior_shapley_covariance
from .updates import sherman_morrison_append
class ActiveGPControlVariate:
 def __init__(self,m,kernel,eta):
  self.m=m;self.kernel=kernel;self.eta=float(eta);self.design=[];self.y=[];self.inverse=np.empty((0,0));self.kphi=np.empty((m,0));self.prior=prior_shapley_covariance(m,kernel);self.alpha=None;self.scale=1.;self.shift=0.
 def append(self,s,y):
  s=np.asarray(s,bool); k=np.array([self.kernel(s,d) for d in self.design]); inv,schur,ok=sherman_morrison_append(self.inverse,k,self.kernel(s,s),self.eta)
  if not ok:return False,schur
  self.inverse=inv;self.design.append(s.copy());self.y.append(float(y));self.kphi=np.column_stack((self.kphi,cross_covariance(s,self.kernel)));return True,schur
 def finalize(self,L,U):
  if not self.design: raise ValueError('empty GP design')
  self.alpha=self.inverse@np.asarray(self.y); p=self.alpha[self.alpha>0].sum(); n=self.alpha[self.alpha<0].sum(); base=self.kernel.sigma0**2; lo=base*(self.kernel.rho**self.m*p+n);hi=base*(p+self.kernel.rho**self.m*n); span=hi-lo
  self.scale=min(1.,(U-L)/span) if span>0 else 1.;self.shift=L-self.scale*lo
  phi=self.scale*self.kphi@self.alpha; post=posterior_shapley_covariance(self.prior,self.kphi,self.inverse)*self.scale**2
  return phi,post,(lo,hi)
 def predict(self,s):
  if self.alpha is None:return 0.
  k=self.kernel.vector(np.asarray(self.design),s);return float(self.shift+self.scale*(k@self.alpha))
 def posterior_attribution_cov(self,s):
  k=self.kernel.vector(np.asarray(self.design),s); return cross_covariance(s,self.kernel)-self.kphi@self.inverse@k
 def acquisition(self,s):
  k=self.kernel.vector(np.asarray(self.design),s); var=self.kernel(s,s)-k@self.inverse@k; c=self.posterior_attribution_cov(s); return float(c@c/(max(var,1e-12)+self.eta**2))
