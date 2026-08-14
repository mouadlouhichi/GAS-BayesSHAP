import numpy as np
from .strata import ResidualRecord
class ResidualStore:
 def __init__(self,m): self.m=m;self.records=[[[] for _ in range(m)] for _ in range(m)]
 def add(self,r): self.records[r.stratum][r.feature].append(r)
 def values(self,s,i): return [r.residual_value for r in self.records[s][i]]
 def count(self,s,i): return len(self.records[s][i])
 def means(self,strict=True):
  out=np.zeros(self.m)
  for i in range(self.m):
   for s in range(self.m):
    a=self.values(s,i)
    if not a and strict: raise ValueError(f'missing stratum ({i},{s})')
    out[i]+=np.mean(a) if a else 0.
  return out/self.m
 def sigma(self,default=.5):
  x=np.zeros((self.m,self.m))
  for s in range(1,self.m-1):
   for i in range(self.m):
    a=self.values(s,i); x[s,i]=np.std(a,ddof=1) if len(a)>1 else default
  return x
 def missing(self): return [(i,s) for i in range(self.m) for s in range(self.m) if not self.records[s][i]]
def sample_round(oracle,x,s,predict,store,iteration,rng):
 """Both Lemma F mechanisms from a single cardinality-q coalition."""
 m=len(s); v=oracle.evaluate(x,s); ms=predict(s); seed=int(rng.integers(2**63-1))
 for i in range(m):
  if not s[i]:
   u=s.copy();u[i]=1; r=(oracle.evaluate(x,u)-v)-(predict(u)-ms);store.add(ResidualRecord(i,int(s.sum()),int(sum(int(z)<<j for j,z in enumerate(s))),'add-one',float(r),iteration,seed))
  elif s.sum()>0:
   u=s.copy();u[i]=0; r=(v-oracle.evaluate(x,u))-(ms-predict(u));store.add(ResidualRecord(i,int(s.sum())-1,int(sum(int(z)<<j for j,z in enumerate(s))),'remove-one',float(r),iteration,seed))
