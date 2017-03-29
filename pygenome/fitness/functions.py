import numpy as np
import pygenome as pg


def onemax(vector):
    '''
    Onemax fitness function

    Args:
        vector (array): binary array
    
    Returns:
        sum of ones in the binary array
    '''
    return np.sum(vector)


def sphere_model(vector):
    '''
    Sphere Model fitness function

    Args:
        vector (array): float array
    
    Returns:
        computed solution
    '''
    sphere = np.vectorize(lambda x : x ** 2.0)

    return np.sum(sphere(vector))
