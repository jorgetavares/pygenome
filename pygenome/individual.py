import numpy as np

class Individual(object):
    '''
    Base class for all type of individuals
    '''

    def __init__(self):
        self.fitness = None # this should be an object since fitness can be more than a simple value
        self.genome = None  # this should be an object that can be of any type 


def linear_chromossome(size, min_value=0, max_value=1):
    '''
    Linear Chromossome

    Args:
        size (int): the fixed size of the chromossome
        min_value (int): minimum value that can be in the chromossome
        max_value (int): maximum value (inclusive) that can be in the chromossome
    
    Returns:
        fixed size chromossome initialized with random integers
    '''
    value_range = (max_value - min_value) + 1
    chromossome = np.random.randint(value_range, size=size) + min_value
    
    return chromossome
