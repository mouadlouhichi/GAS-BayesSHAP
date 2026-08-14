import numpy as np
def errors(estimate,truth):
 e=np.asarray(estimate)-np.asarray(truth);return {'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),'max_error':float(np.max(abs(e)))}
