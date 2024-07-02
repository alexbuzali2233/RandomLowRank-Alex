import numpy as np
def coherence(V,k):
    rowNorms = []
    for i in range(len(V)):
        rowNorms.append(np.linalg.norm(V[i,:k]))
    return max(rowNorms)

def klDivergence(dist,p,q):

    sum = 0

    for x in dist:
        if p[x] != 0 and q[x] != 0:
            sum += p[x] * np.log(p[x]/q[x])

    return sum

def matrixLeverageScores(V, type = 'row'):
    """
    Gets leverage scores for each row/column of V. eturn a distribution of 
    with element i reperesenting the leverage score of the ith row/column.
    """

    q = []

    if type == 'row':
        for i in range(len(V)):
            squareNorm = np.linalg.norm(V[i,:])**2
            q.append(squareNorm)
    elif type == 'column':
        for j in range(len(V[0])):
            squareNorm = np.linalg.norm(V[:,j])**2
            q.append(squareNorm)

    return np.array(q)