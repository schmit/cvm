'''
Contains standard kernels
'''

import numpy as np

import scipy.spatial.distance as dist


def rbf_kernel(X, Y, gamma=1.0):
    '''
    RBF kernel
    '''
    return np.exp(-gamma * dist.cdist(X, Y)**2)

def poly_kernel(X, Y, degree=2):
    '''
    Polynomial kernel
    '''
    return (np.dot(X, Y.T)+1)**degree
