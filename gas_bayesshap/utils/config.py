import hashlib,json,yaml
from pathlib import Path
def load(path):
 d=yaml.safe_load(Path(path).read_text());validate(d);return d
def validate(c):
 for k in ['M','sigma0','lengthscale','eta','epsilon','delta','max_budget','max_rounds']:
  if k not in c:raise ValueError('missing config '+k)
 if c['M']<1 or c['eta']<=0 or c['lengthscale']<=0 or not 0<c['delta']<1 or c['max_budget']<0:raise ValueError('invalid config')
def digest(c):return hashlib.sha256(json.dumps(c,sort_keys=True,default=str).encode()).hexdigest()
