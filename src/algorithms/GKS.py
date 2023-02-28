import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds
import scipy as sp
import time

def GKS(A, k, timed = False, sparseSVD = False):

    t0 = time.time()

    if sparseSVD:
        VT = svds(A, k, solver='lobpcg')
        V = VT.T
    else:
        VT = svd(A)[2]
        V = VT.T[:,:k]

    qrobj = sp.linalg.qr(V.T,pivoting=True)
    pi = qrobj[2][:k]

    Q2 = np.linalg.qr(A[:,pi])[0]

    Ahat = Q2@Q2.T@A

    if timed:
        t1 = time.time()
        return Ahat, t1-t0

    return Ahat