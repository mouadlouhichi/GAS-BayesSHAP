import numpy as np
from gas_bayesshap.gp.updates import sherman_morrison_append
def test_sherman_morrison_matches_direct():
 eta=.1;K=np.array([[1.]]) ;inv=np.linalg.inv(K+eta**2*np.eye(1));k=np.array([.3]);got,_,ok=sherman_morrison_append(inv,k,1.,eta);direct=np.linalg.inv(np.array([[1+eta**2,.3],[.3,1+eta**2]]));assert ok and np.allclose(got,direct)
def test_near_duplicate_is_rejected():
 inv=np.array([[1/.01]]);_,_,ok=sherman_morrison_append(inv,np.array([1.]),1.,.1);assert not ok
