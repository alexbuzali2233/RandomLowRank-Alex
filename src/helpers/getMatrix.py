import numpy as np
from numpy.linalg import qr
from scipy.special import erfc
from scipy.linalg import hadamard
from ..config import *

class spectrumObject():

    def __init__(self, levels, decayRange, decayTypes):
        self.levels = levels
        self.decayRange = decayRange
        self.decayTypes = decayTypes

def getMatrix(dim, rowSpace, spectrum, returnSVD = False, coherenceScalar = .1):
    
    #Check for valid inputs
    assert type(dim) == int and dim > 0, 'Matrix dimension must be a positive '\
    'integer.'

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
    # if spectrum == 'smooth gap':
    #     decayLength = int(np.floor(.7*k))
    #     x = np.linspace(0, 1, dim)
    #     x *= 5/(x[k-1] - x[k-1-decayLength])
    #     x += 2.5 - x[k-1]
    #
    #     singularValues = .5*(1+erfc(steepness*x))/1.5
    #     beta = np.log(res)/np.log(singularValues[k])
    #     singularValues **= beta
    #     sigma = np.diag(singularValues)

    levels = spectrum.levels
    ranges = spectrum.decayRange
    types = spectrum.decayTypes
    singularValues = np.zeros(dim)

    for typeIndex in range(len(types)):
        levelIndex = typeIndex
        rangeIndex = 2*typeIndex

        #First flat section
        singularValues[0:ranges[rangeIndex]] = levels[levelIndex] * np.ones(ranges[rangeIndex])

        #Decay region
        numDecayPoints = ranges[rangeIndex + 1] - ranges[rangeIndex] + 1
        if types[typeIndex] == 'smoothgap':
            t = np.linspace(-2, 2, numDecayPoints)
            decayCurve = .5 * (erfc(t) + 1)
            beta = np.log(levels[levelIndex]/levels[levelIndex + 1])/np.log(2.981)
            decayCurve **= beta
            decayCurve *= levels[levelIndex]/decayCurve[0]

        singularValues[ranges[rangeIndex]:ranges[rangeIndex + 1] + 1] = decayCurve

        #Last flat section
        lastIndexSet = ranges[rangeIndex + 1]
        singularValues[lastIndexSet:] = levels[-1]*np.ones(dim - lastIndexSet)

        sigma = np.diag(singularValues)

    if returnSVD:
        return U, sigma, V
    else:
        return U @ sigma@ V.T