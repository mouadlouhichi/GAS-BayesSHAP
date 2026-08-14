import numpy as np
from scipy.optimize import minimize
def coupled_neyman(sigma,budget=1.):
 """Theorem A coupled convex program; q=1,...,M-1 is retained."""
 M=sigma.shape[0]; probs=np.zeros(M)
 if M<=2:return probs,{'success':True,'message':'no interior strata','objective':0.,'counts':probs}
 A=np.array([np.sum(sigma[s]**2) for s in range(M)])
 def obj(x):
  return sum(A[s]/max((M-s)*x[s-1]+(s+1)*x[s],1e-15) for s in range(1,M-1))/M
 res=minimize(obj,np.full(M-1,budget/(M-1)),method='SLSQP',bounds=[(0,None)]*(M-1),constraints={'type':'eq','fun':lambda x:x.sum()-budget})
 if not res.success or not np.all(np.isfinite(res.x)):
  probs[1:]=1/(M-1); return probs,{'success':False,'message':res.message,'objective':obj(probs[1:]),'counts':probs*budget}
 probs[1:]=res.x/res.x.sum(); return probs,{'success':True,'message':res.message,'objective':float(res.fun),'counts':res.x}
