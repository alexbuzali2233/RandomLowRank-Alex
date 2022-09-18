import numpy as np
from numpy.linalg import qr
from sketch import sketch

def randomSVD(A, k, p, power, sketchType):

    assert type(power) == int and power >= 0, 'Power iteration parameter must be'\
        ' a nonnegative integer.'

    assert type(k) == int and k > 0 and k <= min(len(A),len(A[0])), 'Target rank'\
        ' must be a positive integer less than min(# rows, #columns)'

    assert type(p) == int and p >= 0, 'Oversampling parameter must be'\
        ' a nonnegative integer.'

    ASketch = sketch(A, k + p, 'right', sketchType)
    Q = qr(ASketch)[0]

    for i in range(power):
        Q=qr((A@A.T)@Q)[0]

    svd = np.linalg.svd(Q.T@A)

    U = Q@svd[0][:,:k]
    sigma = svd[1][:k,:k]
    V = svd[2][:,:k]

    return U,sigma,V

