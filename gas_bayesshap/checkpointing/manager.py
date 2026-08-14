import os,json,pickle,tempfile,hashlib
from pathlib import Path
class CheckpointManager:
 def __init__(self,root,run_id): self.root=Path(root)/run_id;self.root.mkdir(parents=True,exist_ok=True);self.manifest=self.root/'checkpoint_manifest.json'
 def save(self,stage,state,iteration,meta):
  name=f'{stage}_{iteration}.pkl'; target=self.root/name
  fd,tmp=tempfile.mkstemp(dir=self.root,prefix='.tmp_')
  try:
   with os.fdopen(fd,'wb') as f: pickle.dump(state,f,protocol=5);f.flush();os.fsync(f.fileno())
   digest=hashlib.sha256(Path(tmp).read_bytes()).hexdigest();os.replace(tmp,target)
   old={}
   if self.manifest.exists():
    try:old=json.loads(self.manifest.read_text())
    except Exception:pass
   d={'latest_valid_checkpoint':name,'previous_valid_checkpoint':old.get('latest_valid_checkpoint'),'stage':stage,'iteration':iteration,'query_count':meta.get('query_count',0),'config_hash':meta.get('config_hash'),'oracle_hash':meta.get('oracle_hash'),'result_hash':digest}
   fd2,tmp2=tempfile.mkstemp(dir=self.root,prefix='.manifest_')
   with os.fdopen(fd2,'w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
   os.replace(tmp2,self.manifest);return target
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
 def load_latest(self):
  if not self.manifest.exists():return None
  d=json.loads(self.manifest.read_text());p=self.root/d['latest_valid_checkpoint']
  if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=d['result_hash']: raise RuntimeError('corrupted checkpoint rejected')
  return d,pickle.loads(p.read_bytes())
