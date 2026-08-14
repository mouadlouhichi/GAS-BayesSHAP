"""Deterministic interventional coalition oracle with auditable accounting."""
from abc import ABC, abstractmethod
import hashlib, numpy as np
from ..game.subsets import bitmask
class CoalitionOracle(ABC):
    @abstractmethod
    def evaluate(self,x,coalition)->float: ...
    @property
    @abstractmethod
    def output_bounds(self): ...
class InterventionalOracle(CoalitionOracle):
    def __init__(self,model,background,output_bounds=None, configuration_hash=''):
        self.model=model; self.background=np.asarray(background,float); self._bounds=output_bounds
        if self.background.ndim != 2: raise ValueError('background must be two dimensional')
        self.num_coalition_evals=self.num_model_evals=0; self.cache={}; self.cache_hits=0; self.configuration_hash=configuration_hash
        self.background_hash=hashlib.sha256(self.background.tobytes()).hexdigest()
    @property
    def output_bounds(self): return self._bounds
    @property
    def oracle_hash(self): return hashlib.sha256((repr(self.model)+self.background_hash+str(self._bounds)).encode()).hexdigest()
    def evaluate(self,x,coalition):
        x=np.asarray(x,float); s=np.asarray(coalition,dtype=bool)
        if x.shape != (self.background.shape[1],) or s.shape != x.shape: raise ValueError('input/coalition dimension mismatch')
        key=(hashlib.sha256(x.tobytes()).hexdigest(),bitmask(s),self.background_hash,self.configuration_hash,self.oracle_hash)
        if key in self.cache: self.cache_hits+=1; return self.cache[key]
        hybrids=np.tile(x,(len(self.background),1)); hybrids[:,~s]=self.background[:,~s]
        vals=np.asarray([self.model(row) for row in hybrids],float)
        if not np.all(np.isfinite(vals)): raise FloatingPointError('oracle returned NaN/Inf')
        self.num_coalition_evals+=1; self.num_model_evals+=len(vals); value=float(vals.mean()); self.cache[key]=value; return value
