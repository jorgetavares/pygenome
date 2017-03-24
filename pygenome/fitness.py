import numpy as np
import pygenome.population as population


class Fitness(object):
    '''
    Base class for a Fitness
    '''

    def __init__(self, value=None):
        self.value = value


def evaluate_population(pop, fitness_fn, **kargs):
    '''
    Evaluate Population

    Args:
        pop (Population): population to be evaluated in-place
        fitness_fn (function): fitness function that receives an individual
        **kargs: keyword arguments that fitness_fn might have

    Returns:
        evaluated population by calling the fitness function
    '''
    for i in range(pop.size):
        ind = pop.individuals[i]
        ind.fitness = Fitness(fitness_fn(ind.genome, **kargs))
    
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
