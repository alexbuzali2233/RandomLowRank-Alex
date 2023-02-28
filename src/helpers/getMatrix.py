import numpy as np
from numpy.linalg import qr
from scipy.special import erfc
from scipy.linalg import hadamard
from ..config import *

def getMatrix(dim, k, res, rowSpace = 'random', spectrum = 'smooth gap',
                returnSVD = False, coherenceScalar = .1, steepness = 1):
    
    #Check for valid inputs
    assert type(dim) == int and dim > 0, 'Matrix dimension must be a positive '\
    'integer.'
    assert type(k) == int and k > 0, 'Target rank must be a positive integer.'
    assert type(dim >= k), 'Target rank cannot excced matrix dimension'
    assert res >0 and res < 1, 'Target residual must be in the open '\
    'interval (0,1).'

    #Constructing the column space (left singular subspace) of the matrix
    U = qr(np.random.normal(size=(dim,dim)))[0]

    #Constructing the row space (right singular subspace) of the matrix
    if rowSpace == 'random':
        V = qr(np.random.normal(size=(dim,dim)))[0]
    elif rowSpace == 'hadamard':

        #Scipy hadamard only works for powers of 2. Can manually save other
        #Hadamard matrices using saveHadamard.jl and then load them here
        try:
            V= hadamard(dim)/np.sqrt(dim)
        except:
            
            a = os.path.abspath(".").rfind('/')
            projectPath = os.path.abspath(".")[:a+1]
            V=np.load(projectPath + hadamardMatricesPath + 'hadamard' + str(dim) + '.npy')/np.sqrt(dim)
    elif rowSpace == 'incoherent':
        try:
            V= hadamard(dim)/np.sqrt(dim)
        except:
            V=np.load(hadamardMatricesPath + 'hadamard' + str(dim) + '.npy')/np.sqrt(dim)
        L, _, R = np.linalg.svd(V +coherenceScalar*np.random.normal(size=(dim,dim)))
        V=L@R
    elif rowSpace == 'permutation':
        V=np.eye(dim)[:,np.random.permutation(dim)]
    elif rowSpace == 'coherent':
        V=np.eye(dim)[:,np.random.permutation(dim)]
        L, _, R = np.linalg.svd(V + coherenceScalar*np.random.normal(size=(dim,dim)))
        V=L@R
    else:
        raise Exception ('Not a valid row space.')
    
    #Constructing the singular spectrum
    if spectrum == 'smooth gap':
        decayLength = int(np.floor(.7*k))
        x = np.linspace(0, 1, dim)
        x *= steepness*5/(x[k-1] - x[k-1-decayLength])
        x += 2.5 - x[k-1]
        
        singularValues = .5*(1+erfc(x))/1.5
        beta = np.log(res)/np.log(singularValues[k])
        singularValues **= beta
        sigma = np.diag(singularValues)

    if returnSVD:
        return U, sigma, V
    else:
        return U @ sigma@ V.T

# U,sigma,V = getMatrix(96,48,1e-12,'incoherent',returnSVD = True)
# k = np.arange(0,96,1)
# plt.plot(k,np.diag(sigma))
# plt.show()