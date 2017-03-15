import numpy as np
import pygenome.population as population


class Fitness(object):
    '''
    Base class for a Fitness
    '''

    def __init__(self, value=None):
        self.value = value


def evaluate_population(pop, fitness_fn):
    '''
    Evaluate population

    Args:
        pop (Population): population to be evaluated in-place
        fitness_fn (function): fitness function that receives an individual

    Returns:
        evaluated population by cal
    '''
    for i in range(pop.size):
        ind = pop.individuals[i]
        ind.fitness = Fitness(fitness_fn(ind.genome))
    
    return pop


def onemax(vector):
    '''
    Onemax fitness function

    Args:
        vector (array): binary array
    
    Returns:
        sum of ones in the binary array
    '''
    return np.sum(vector)
