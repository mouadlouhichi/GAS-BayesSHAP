import platform,sys,numpy as np
def environment():return {'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}
