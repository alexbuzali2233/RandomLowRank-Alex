import numpy as np
import scipy as sp
from ..helpers.sketch import sketch
from numpy.linalg import svd

def randomGKS(A, k, p, sketchType, twoPass = True):

    if twoPass:

        ASketch = sketch(A,k+p,'right',sketchType)
        Q = np.linalg.qr(ASketch)[0]
        VT = svd(Q.T@A)[2]

    else:

        Z = sketch(A,k+p,'left',sketchType)
        VT = svd(Z)[2]
    
    V = VT.T[:,:k]

    qrobj = sp.linalg.qr(V.T,pivoting=True)
    pi = qrobj[2][:k]

    Q2 = np.linalg.qr(A[:,pi])[0]

    C = A[:,pi]
    Cp = np.linalg.pinv(C)


    #return C@Cp@A
    return Q2@Q2.T@A