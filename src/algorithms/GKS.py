import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds
import scipy as sp
import time

def GKS(A, k, timed = False, sparseSVD = False, getDist = False):
    """
    Golub-Klema-Stewart algorithm as outlined in Algorithm 4.1. Returns
    matrix approximation Â = B_1 * (B_2)^T.

    Parameters:
    -A (ndarray): matrix to be approximated
    -k (int): approximation rank
    -timed (bool): if True, algorithm returns run time in addition to Â
    -sparseSVD (bool): if True, computes partial SVD using sparse methods
    -getDist (bool): if True, returns permutation matrix Π in addition to Â
    """

    t0 = time.time()

    # Step 1
    if sparseSVD:
        VT = svds(A, k, solver='lobpcg')
        V = VT.T
    else:
        VT = svd(A)[2]
        V = VT.T[:,:k]

    # Step 2
    qrobj = sp.linalg.qr(V.T,pivoting=True)
    pi = qrobj[2][:k]

    # Step 3
    B1 = np.linalg.qr(A[:,pi])[0]
    B2T = B1.T@A
    Ahat = B1@B2T

    t1 = time.time()

    # Return
    if getDist and timed:
        return Ahat, pi, t1-t0

    elif getDist:
        return Ahat, pi

    elif timed:
        return Ahat, t1-t0

    return Ahat