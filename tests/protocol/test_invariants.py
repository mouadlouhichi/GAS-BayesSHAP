import numpy as np
from gas_bayesshap.game.oracle import InterventionalOracle
from gas_bayesshap.residual.neyman import coupled_neyman
def test_oracle_cache_and_accounting():
 o=InterventionalOracle(lambda x:x.sum(),np.zeros((3,2)),(-1,1));x=np.ones(2);s=np.array([1,0],bool);assert o.evaluate(x,s)==1;assert o.num_coalition_evals==1; o.evaluate(x,s);assert o.num_coalition_evals==1
def test_extremes_zero_allocation():
 p,_=coupled_neyman(np.ones((5,5)));assert p[0]==0
