"""Declared domain games; all are deterministic once model/background are fixed."""
import numpy as np
from .oracle import InterventionalOracle
class MembershipGame(InterventionalOracle):
 def __init__(self,model,background):super().__init__(model,background,(0.,1.))
class ContrastiveGame(InterventionalOracle):
 def __init__(self,model_c,model_other,background):super().__init__(lambda x:float(model_c(x)-model_other(x)),background,(-1.,1.))
class GlobalArchetypeGame(InterventionalOracle):
 def __init__(self,model,archetypes,background):
  self.archetypes=np.asarray(archetypes);super().__init__(model,background,(0.,1.))
 def evaluate(self,x,coalition):
  # x is intentionally ignored: global average over frozen archetypes.
  return float(np.mean([super(GlobalArchetypeGame,self).evaluate(a,coalition) for a in self.archetypes]))
class SilhouetteGame:
 def __init__(self,X,clusterer):self.X=np.asarray(X);self.clusterer=clusterer;self.num_coalition_evals=self.num_model_evals=0;self._bounds=(-1.,1.)
 @property
 def output_bounds(self):return self._bounds
 @property
 def oracle_hash(self):return 'silhouette-'+str(self.X.shape)
 @property
 def background_hash(self):return 'none'
 def evaluate(self,x,coalition):
  from sklearn.metrics import silhouette_score
  s=np.asarray(coalition,bool);self.num_coalition_evals+=1
  if not s.any():return 0.
  labels=self.clusterer.fit_predict(self.X[:,s]);self.num_model_evals+=1
  return float(silhouette_score(self.X[:,s],labels)) if len(set(labels))>1 else 0.
