import numpy as np

def mask_from_bits(bits, m=None):
    a=np.asarray(bits,dtype=bool)
    return a if m is None else a.reshape(m)
def bitmask(mask):
    return sum(int(v)<<i for i,v in enumerate(np.asarray(mask,dtype=bool)))
def from_bitmask(value,m): return np.array([(value>>i)&1 for i in range(m)],dtype=bool)
def all_coalitions(m):
    for x in range(1<<m): yield from_bitmask(x,m)
def random_coalition(rng,m,size=None):
    q=int(rng.integers(m+1) if size is None else size); out=np.zeros(m,dtype=bool); out[rng.permutation(m)[:q]]=True; return out
