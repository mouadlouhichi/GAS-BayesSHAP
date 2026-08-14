import numpy as np
def project(raw,delta_total,variances):
 v=np.maximum(np.asarray(variances,float),0); total=v.sum()
 if total<=0: raise FloatingPointError('posterior diagonal variance sum is zero')
 return np.asarray(raw)+v*(delta_total-np.sum(raw))/total
def projected_widths(raw_widths,variances):
 v=np.asarray(variances,float);return np.asarray(raw_widths)+v/v.sum()*np.sum(raw_widths)
