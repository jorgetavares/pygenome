import numpy as np

class Individual(object):
    '''
    Base class for all type of individuals
    '''

    def __init__(self):
        self.fitness = None # this should be an object since fitness can be more than a simple value
        self.genome = None  # this should be an object that can be of any type 


def integer_chromossome(size, min_value=0, max_value=1):
    '''
    Integer Chromossome (random)

    Args:
        size (int): the fixed size of the chromossome
        min_value (int): minimum value that can be in the chromossome
        max_value (int): maximum value (exclusive) that can be in the chromossome
    
    Returns:
        fixed size chromossome initialized with random integers
    '''
    value_range = (max_value - min_value)
    chromossome = np.random.randint(value_range, size=size) + min_value
    
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


def uniform_chromossome(size, min_value=0.0, max_value=1.0)):
    '''
    Uniform Chromossome (random)

    Args:
        size (int): the fixed size of the chromossome
        min_value (float): minimum value that can be in the chromossome
        max_value (float): maximum value (exclusive) that can be in the chromossome
    
    Returns:
        fixed size chromossome initialized with random floats using a uniform distribution
    '''
    return np.random.uniform(min_value, max_value, size)
    

def normal_chromossome(size, mean=0.0, sigma=1.0)):
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
