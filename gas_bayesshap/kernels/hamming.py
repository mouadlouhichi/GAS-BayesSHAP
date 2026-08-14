import numpy as np
class ExponentialHammingKernel:
    def __init__(self,sigma0=1.,lengthscale=1.5):
        if sigma0 <= 0 or lengthscale <= 0: raise ValueError('sigma0 and lengthscale must be positive')
        self.sigma0=float(sigma0); self.lengthscale=float(lengthscale); self.rho=float(np.exp(-1./lengthscale))
    def __call__(self,s,t): return self.sigma0**2 * self.rho**int(np.count_nonzero(np.asarray(s,dtype=bool)!=np.asarray(t,dtype=bool)))
    def vector(self, design, s):
        d=np.asarray(design,dtype=bool); return self.sigma0**2*self.rho**np.count_nonzero(d!=np.asarray(s,dtype=bool),axis=1)
