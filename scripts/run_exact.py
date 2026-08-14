import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from gas_bayesshap.game.shapley import exact_shapley
from gas_bayesshap.game.oracle import InterventionalOracle
m=4;r=np.random.default_rng(1);o=InterventionalOracle(lambda x:1/(1+np.exp(-x.sum())),r.normal(size=(10,m)),(0,1));x=r.normal(size=m);print(exact_shapley(lambda s:o.evaluate(x,s),m))
