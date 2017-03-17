import numpy as np
from copy import deepcopy

class Individual(object):
    '''
    Base class for all type of individuals
    '''

    def __init__(self, fitness=None, genome=None):
        self.fitness = fitness # this should be an object since fitness can be more than a simple value
        self.genome = genome   # this should be an object that can be of any type 

    def clone(self):
        return deepcopy(self)
        

def integer_chromossome(size, low=0, high=1):
    '''
    Integer Chromossome (random)

    Args:
        size (int): the fixed size of the chromossome
        min_value (int): minimum value that can be in the chromossome
        max_value (int): maximum value (inclusive) that can be in the chromossome
    
    Returns:
        fixed size chromossome initialized with random integers
    '''
    high_value = high + 1
    chromossome = np.random.randint(low, high=high_value, size=size)
    
    return chromossome


def permutation_chromossome(size):
    '''
    Permutation Chromossome (random)

    Args:
        size (int): the fixed size of the chromossome
    
    Returns:
        fixed size chromossome initialized with a random permutation from 0 to size -1 
    '''
    return np.random.permutation(size)


def uniform_chromossome(size, low=0.0, high=1.0):
    '''
    Uniform Chromossome (random)

    Args:
        size (int): the fixed size of the chromossome
        min_value (float): minimum value that can be in the chromossome
        max_value (float): maximum value (exclusive) that can be in the chromossome
    
    Returns:
        fixed size chromossome initialized with random floats using a uniform distribution
    '''
    return np.random.uniform(low=low, high=high, size=size)
    

def normal_chromossome(size, mean=0.0, sigma=1.0):
    '''
    Normal Chromossome (random)

    Args:
        size (int): the fixed size of the chromossome
        mean (float): mean of the normal distribution
        sigma (float): sigma of the normal distribution
    
    Returns:
        fixed size chromossome initialized with random floats using a normal distribution
    '''
    return np.random.normal(loc=mean, scale=sigma, size=size)
