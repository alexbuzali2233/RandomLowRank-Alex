import numpy as np

def coherence(V,k):
    rowNorms = []
    for i in range(len(V)):
        rowNorms.append(np.linalg.norm(V[i,:k]))
    return max(rowNorms)
        