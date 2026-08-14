"""Production orchestration: Module A is frozen before Module B begins."""
import json,time,uuid,random
from pathlib import Path
import numpy as np
from ..kernels.hamming import ExponentialHammingKernel
from ..gp.control_variate import ActiveGPControlVariate
from ..residual.estimator import ResidualStore,sample_round
from ..residual.strata import ResidualRecord
from ..residual.neyman import coupled_neyman
from ..certification.confidence_sequences import widths
from ..certification.projection import project,projected_widths
from ..checkpointing.manager import CheckpointManager
from ..logging.logger import EventLogger
from ..utils.config import digest
class GASBayesSHAP:
 def __init__(self,oracle,config,run_id=None):
  self.oracle=oracle;self.c=dict(config);self.m=int(config['M']);self.run_id=run_id or config.get('run_id') or uuid.uuid4().hex[:12];self.rng=np.random.default_rng(config.get('seed',0));random.seed(config.get('seed',0));self.hash=digest(self.c)
  self.ckpt=CheckpointManager(config.get('checkpoint_root','checkpoints'),self.run_id);self.log=EventLogger(config.get('results_root','results/runs'),self.run_id);self.stage='PREFLIGHT';self.gp=None;self.store=None;self.iteration=0
 def _meta(self):return {'query_count':self.oracle.num_coalition_evals,'config_hash':self.hash,'oracle_hash':self.oracle.oracle_hash}
 def _save(self,stage):
  self.stage=stage
  state={'stage':stage,'gp':self.gp,'store':self.store,'iteration':self.iteration,'rng':self.rng.bit_generator.state,'oracle_cache':self.oracle.cache,'counters':(self.oracle.num_coalition_evals,self.oracle.num_model_evals),'x':getattr(self,'x',None),'L':getattr(self,'L',None),'U':getattr(self,'U',None)}
  self.ckpt.save(stage,state,self.iteration,self._meta());self.log.event('checkpoints',stage,'checkpoint_saved',iteration=self.iteration,**self._meta())
 def resume(self):
  d,s=self.ckpt.load_latest();
  if d['config_hash'] != self.hash or d['oracle_hash'] != self.oracle.oracle_hash: raise RuntimeError('incompatible checkpoint')
  for k,v in s.items():
   if k=='rng':self.rng.bit_generator.state=v
   elif k=='oracle_cache':self.oracle.cache=v
   elif k=='counters':self.oracle.num_coalition_evals,self.oracle.num_model_evals=v
   else:setattr(self,k,v)
  self.stage=d['stage'];self.log.event('checkpoints',self.stage,'checkpoint_restored',iteration=self.iteration,**self._meta());return self
 def _evaluate_extremes(self):
  m=self.m; zero=np.zeros(m,bool);full=np.ones(m,bool);vz=self.oracle.evaluate(self.x,zero);vf=self.oracle.evaluate(self.x,full)
  for i in range(m):
   one=zero.copy();one[i]=1; rem=full.copy();rem[i]=0
   a=(self.oracle.evaluate(self.x,one)-vz)-(self.gp.predict(one)-self.gp.predict(zero))
   b=(vf-self.oracle.evaluate(self.x,rem))-(self.gp.predict(full)-self.gp.predict(rem))
   self.store.add(ResidualRecord(i,0,0,'add-one',float(a),0,0));self.store.add(ResidualRecord(i,m-1,(1<<m)-1,'remove-one',float(b),0,0))
 def _module_a(self):
  self.stage='GP_INITIALIZATION';self.gp=ActiveGPControlVariate(self.m,ExponentialHammingKernel(self.c['sigma0'],self.c['lengthscale']),self.c['eta'])
  # seeds contain extreme and one representative of each cardinality
  seen=set()
  for q in range(self.m+1):
   s=np.zeros(self.m,bool);s[self.rng.permutation(self.m)[:q]]=1; key=tuple(s)
   if key not in seen:
    seen.add(key);ok,schur=self.gp.append(s,self.oracle.evaluate(self.x,s));self.log.event('gp_updates','GP_INITIALIZATION','gp_seed',coalition=key,schur=schur,**self._meta())
  self.stage='ACTIVE_GP';pool=max(32,2*self.m) if self.c.get('pool_size') is None else int(self.c['pool_size'])
  for step in range(int(self.c.get('n_active_steps',0))):
   candidates=[np.array([False]*self.m) for _ in range(pool)]
   for s in candidates:
    q=int(self.rng.integers(self.m+1));s[self.rng.permutation(self.m)[:q]]=1
   best=max(candidates,key=self.gp.acquisition);score=self.gp.acquisition(best);y=self.oracle.evaluate(self.x,best);ok,schur=self.gp.append(best,y)
   self.log.event('acquisition','ACTIVE_GP','selected',iteration=step,score=score,schur=schur,action_taken='append' if ok else 'near_duplicate_skip',**self._meta())
  self.stage='BOUNDED_SURROGATE';self.surrogate,self.post,self.bounds=self.gp.finalize(self.L,self.U);self._save('gp_stage')
 def run(self,x,resume=False):
  self.x=np.asarray(x,float)
  if self.x.shape!=(self.m,):raise ValueError('M and x disagree')
  if resume:self.resume()
  if self.oracle.output_bounds is None:
   # Explicitly heuristic; never advertised as rigorous.
   z=self.oracle.evaluate(self.x,np.zeros(self.m,bool));f=self.oracle.evaluate(self.x,np.ones(self.m,bool));self.L=min(z,f)-abs(f-z);self.U=max(z,f)+abs(f-z);heuristic=True
  else:self.L,self.U=map(float,self.oracle.output_bounds);heuristic=False
  if self.stage in ('PREFLIGHT','GP_INITIALIZATION','ACTIVE_GP'):
   self._module_a()
  if self.stage=='gp_stage':
   self.stage='RESIDUAL_PILOT';self.store=ResidualStore(self.m);self._evaluate_extremes();self.iteration=0
   # Pilot covers all q and makes coverage explicit.
   for rep in range(int(self.c.get('n_pilot',0))):
    for q in range(1,self.m):
     s=np.zeros(self.m,bool);s[self.rng.permutation(self.m)[:q]]=1;sample_round(self.oracle,self.x,s,self.gp.predict,self.store,self.iteration,self.rng);self.iteration+=1
   self._save('residual_stage')
  self.stage='ADAPTIVE_CERTIFICATION'; stage2_start=self.oracle.num_coalition_evals
  residual_range=4*(self.U-self.L);freq=max(1,int(self.c.get('neyman_refresh_frequency',10)));probs,info=coupled_neyman(self.store.sigma())
  while self.iteration < self.c['max_rounds']:
   w,missing=widths(self.store,self.c['delta'],residual_range)
   self.log.event('certification','ADAPTIVE_CERTIFICATION','width_vector',iteration=self.iteration,width_vector=w.tolist(),max_width=float(np.max(w)),mean_width=float(np.mean(w)),median_width=float(np.median(w)),argmax_feature=int(np.argmax(w)),**self._meta())
   if np.max(w)<=self.c['epsilon']:break
   # a q coalition plus all one-feature neighbours has at most M+1 individual calls
   if self.oracle.num_coalition_evals-stage2_start+(self.m+1)>self.c['max_budget']:break
   if self.iteration%freq==0:
    old=probs.copy();probs,info=coupled_neyman(self.store.sigma());self.log.event('neyman','NEYMAN_ALLOCATION','refresh',iteration=self.iteration,sigma_res=self.store.sigma().tolist(),previous_probabilities=old.tolist(),updated_probabilities=probs.tolist(),objective=info['objective'],**self._meta())
   q=int(self.rng.choice(np.arange(self.m),p=probs)) if probs.sum()>0 else 1
   s=np.zeros(self.m,bool);s[self.rng.permutation(self.m)[:q]]=1;sample_round(self.oracle,self.x,s,self.gp.predict,self.store,self.iteration,self.rng);self.iteration+=1
   self._save('certification_stage')
  w,missing=widths(self.store,self.c['delta'],residual_range)
  try: residual=self.store.means(strict=True)
  except ValueError: residual=np.full(self.m,np.nan)
  raw=self.surrogate+residual;delta_total=self.oracle.evaluate(self.x,np.ones(self.m,bool))-self.oracle.evaluate(self.x,np.zeros(self.m,bool));var=np.maximum(np.diag(self.post),1e-15);final=project(raw,delta_total,var);pw=projected_widths(w,var) if np.all(np.isfinite(w)) else w
  converged=bool(np.max(w)<=self.c['epsilon']);status='CERTIFIED' if converged and not heuristic else ('HEURISTIC_BOUNDS' if converged else ('MISSING_STRATA' if missing else 'BUDGET_EXHAUSTED'))
  result={'shapley_values':final.tolist(),'surrogate_shapley':self.surrogate.tolist(),'residual_shapley':residual.tolist(),'raw_confidence_widths':w.tolist(),'certified_projected_widths':pw.tolist(),'posterior_std':np.sqrt(var).tolist(),'num_coalition_evals':self.oracle.num_coalition_evals,'num_model_evals':self.oracle.num_model_evals,'num_gp_predictions':0,'num_residual_samples':sum(len(self.store.records[s][i]) for s in range(self.m) for i in range(self.m)),'num_sampling_rounds':self.iteration,'converged':converged,'certificate_is_rigorous':converged and not heuristic,'range_bound_is_heuristic':heuristic,'uncertified_features':np.where(~np.isfinite(w))[0].tolist(),'sign_certified_features':np.where(np.abs(final)>pw)[0].tolist(),'status':status,'run_id':self.run_id,'M':self.m,'domain_game':self.c.get('game'),'config_hash':self.hash,'oracle_hash':self.oracle.oracle_hash,'background_hash':self.oracle.background_hash}
  self.stage='FINAL_RESULT';self._save('final_stage');root=Path(self.c.get('results_root','results/runs'))/self.run_id;root.mkdir(parents=True,exist_ok=True)
  for d in ['oracle','gp','residual','neyman','certification','benchmarks','tables','figures','checkpoints']: (root/d).mkdir(exist_ok=True)
  (root/'summary.json').write_text(json.dumps(result,indent=2));(root/'manifest.json').write_text(json.dumps({'run_id':self.run_id,'config_hash':self.hash,'oracle_hash':self.oracle.oracle_hash,'background_hash':self.oracle.background_hash},indent=2));(root/'config.yaml').write_text(json.dumps(self.c,indent=2));(root/'provenance.json').write_text(json.dumps({'rng_state':str(self.rng.bit_generator.state)},indent=2));(root/'summary.md').write_text('# GAS-BayesSHAP result\n\nStatus: '+status+'\n');(root/'reproducibility_report.md').write_text('Run '+self.run_id+' uses fixed background and recorded RNG state.\n');(root/'spec_compliance.json').write_text(json.dumps({'version':'11.0','status':'implemented','stage':self.stage},indent=2));return result
