import argparse,numpy as np
from gas_bayesshap import GASBayesSHAP,InterventionalOracle
from gas_bayesshap.utils.config import load
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml');p.add_argument('--resume',action='store_true');p.add_argument('--run-id');p.add_argument('--M',type=int);p.add_argument('--epsilon',type=float);p.add_argument('--delta',type=float);p.add_argument('--max-budget',type=int);p.add_argument('--max-rounds',type=int);p.add_argument('--dataset');p.add_argument('--game');p.add_argument('--status',action='store_true');p.add_argument('--from-stage');p.add_argument('--until-stage');a=p.parse_args();c=load(a.config)
for k in ['M','epsilon','delta','max_budget','max_rounds','game']:
 v=getattr(a,k.replace('_','-'),None) if False else getattr(a,k,None)
 if v is not None:c[k]=v
if a.run_id:c['run_id']=a.run_id
rng=np.random.default_rng(c['seed']);bg=rng.normal(size=(20,c['M']));x=rng.normal(size=c['M']);model=lambda z:float(1/(1+np.exp(-np.sum(z))))
o=InterventionalOracle(model,bg,tuple(c['output_bounds']) if c.get('output_bounds') else None);e=GASBayesSHAP(o,c,a.run_id)
if a.status:
 print(e.ckpt.load_latest()[0] if e.ckpt.manifest.exists() else 'no checkpoint')
else:print(e.run(x,a.resume))
