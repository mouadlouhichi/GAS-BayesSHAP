import numpy as np
def widths(store,delta,residual_range):
 m=store.m; out=np.zeros(m); missing=[]
 for i in range(m):
  for s in range(1,m-1):
   a=np.asarray(store.values(s,i));n=len(a)
   if n<2: out[i]=np.inf;missing.append((i,s));break
   var=float(np.var(a,ddof=1)); log=np.log(np.pi**2*m*m*n*n/(3*delta)); out[i]+=(np.sqrt(2*var*log/n)+7*residual_range*log/(3*(n-1)))/m
 return out,missing
