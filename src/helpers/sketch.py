import numpy as np

def sketch(A, l, side, sketchType):
    assert type(l) == int and l >0, 'Target sketch dimension must be a positive integer.'

    if side == 'left':
        assert l <= len(A), 'Row dimension of sketch matrix cannot exceed'\
            ' row dimension of A'
        if sketchType == 'Gaussian':
            return np.random.normal(size=(l,len(A))) @ A
        
    elif side == 'right':
        assert l <= len(A[0]), 'Column dimension of sketch matrix cannot exceed'\
            'column dimension of A'
        if sketchType == 'Gaussian':
            return A@np.random.normal(size=(len(A[0]),l))