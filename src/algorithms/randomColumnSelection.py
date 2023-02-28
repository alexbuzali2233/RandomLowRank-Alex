import numpy as np
from numpy.linalg import qr, svd

def randomColumnSelection(A, k, p):

    I = np.random.permutation(len(A))[:k+p]
    Q = qr(A[:,I])[0]
    U,singVals,VT = svd(Q.T@A)
    V = VT.T
    sigma = np.diag(singVals)

    return Q @ U[:,:k] @ sigma[:k,:k] @ (V[:,:k]).T