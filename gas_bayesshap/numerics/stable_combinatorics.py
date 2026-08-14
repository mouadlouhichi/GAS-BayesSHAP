"""Integer-safe combinatorics used by the closed-form covariance identities."""
from math import comb, factorial

def choose(n:int,k:int)->int:
    return comb(n,k) if 0 <= k <= n and n >= 0 else 0

def shapley_weight(m:int,s:int)->float:
    return factorial(s)*factorial(m-1-s)/factorial(m)
