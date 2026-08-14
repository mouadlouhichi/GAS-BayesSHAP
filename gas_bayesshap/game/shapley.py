import numpy as np
from .subsets import all_coalitions
from ..numerics.stable_combinatorics import shapley_weight
def exact_shapley(value,m):
    ans=np.zeros(m)
    for s in all_coalitions(m):
        q=int(s.sum())
        for i in range(m):
            if not s[i]:
                u=s.copy();u[i]=1; ans[i]+=shapley_weight(m,q)*(value(u)-value(s))
    return ans
