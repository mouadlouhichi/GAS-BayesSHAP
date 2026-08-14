import json,time
from pathlib import Path
class EventLogger:
 def __init__(self,root,run_id): self.root=Path(root)/run_id/'logs';self.root.mkdir(parents=True,exist_ok=True);self.run_id=run_id
 def event(self,file,stage,event,status='OK',iteration=0,**kw):
  d={'timestamp':time.time(),'run_id':self.run_id,'stage':stage,'iteration':iteration,'event':event,'status':status,**kw}
  with open(self.root/f'{file}.jsonl','a') as f:f.write(json.dumps(d,default=str)+'\n')
  with open(self.root/'run.log','a') as f:f.write(json.dumps(d,default=str)+'\n')
