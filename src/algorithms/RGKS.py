import numpy as np
import scipy as sp
from ..helpers.sketch import sketch
from numpy.linalg import svd

def randomGKS(A, k, p, sketchType, twoPass = True, getDist = False, getV = False, getVAndPi = False):

    if twoPass:

        ASketch = sketch(A,k+p,'right',sketchType)
        Q = np.linalg.qr(ASketch)[0]
        VT = svd(Q.T@A)[2]

    else:

        Z = sketch(A,k+p,'left',sketchType)
        VT = svd(Z)[2]

    V = VT.T[:,:k]

    qrobj = sp.linalg.qr(V.T,pivoting=True)
    pi = qrobj[2]

    if getVAndPi:
        return V, pi

    pi = qrobj[2][:k]
    Q2 = np.linalg.qr(A[:,pi])[0]

    if getDist and getV:
        return Q2@Q2.T@A, pi, V

    if getDist:
        return Q2@Q2.T@A, pi

    return Q2@Q2.T@A