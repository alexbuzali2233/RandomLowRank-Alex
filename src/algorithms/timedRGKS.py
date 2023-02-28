import numpy as np
import scipy as sp
from ..helpers.sketch import sketch
from numpy.linalg import svd
import time

def timedRGKS(A, k, p, sketchType, twoPass = True):

    t0 = time.time()

    if twoPass:

        ASketch = sketch(A,k+p,'right',sketchType)
        t1 = time.time()
        Q = np.linalg.qr(ASketch)[0]
        Z = Q.T@A
        t2 = time.time()

    else:
        Z = sketch(A,k+p,'left',sketchType)
        
    VT = svd(Z)[2]
    V = VT.T[:,:k]

    qrobj = sp.linalg.qr(V.T,pivoting=True)
    pi = qrobj[2][:k]

    Q2 = np.linalg.qr(A[:,pi])[0]

    Ahat = Q2@Q2.T@A

    t3 = time.time()

    secondPassTime = t2-t1 if twoPass else 0
    totalTime = t3-t0

    return Ahat, secondPassTime, totalTime