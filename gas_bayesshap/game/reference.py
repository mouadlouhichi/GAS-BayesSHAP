"""Independent exponential-time reference engine, only for validation."""
import numpy as np
from .subsets import all_coalitions
from ..numerics.stable_combinatorics import shapley_weight

def cross_covariance_bruteforce(mask,kernel):
 m=len(mask);out=np.zeros(m)
 for i in range(m):
  for s in all_coalitions(m):
   if not s[i]:
    u=s.copy();u[i]=1; out[i]+=shapley_weight(m,int(s.sum()))*(kernel(u,mask)-kernel(s,mask))
 return out

def covariance_bruteforce(m,kernel):
 out=np.zeros((m,m))
 for i in range(m):
  for j in range(m):
   for s in all_coalitions(m):
    if s[i]: continue
    u=s.copy();u[i]=1; ws=shapley_weight(m,int(s.sum()))
    for t in all_coalitions(m):
     if t[j]: continue
     v=t.copy();v[j]=1; wt=shapley_weight(m,int(t.sum()))
     out[i,j]+=ws*wt*(kernel(u,v)-kernel(u,t)-kernel(s,v)+kernel(s,t))
 return out
