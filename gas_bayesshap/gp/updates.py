import numpy as np
def sherman_morrison_append(inv,k,kself,eta):
 if inv.size==0: return np.array([[1/(kself+eta**2)]]), float(kself+eta**2), True
 v=inv@k; schur=float(kself+eta**2-k@v)
 if not np.isfinite(schur) or schur < eta**2: return inv,schur,False
 z=1/schur
 return np.block([[inv+z*np.outer(v,v),(-z*v)[:,None]],[(-z*v)[None,:],np.array([[z]])]]),schur,True
