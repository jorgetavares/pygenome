import numpy as np
import pygenome as pg


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


def evaluate_tree_population(pop, fitness_fn, pset, **kargs):
    '''
    Evaluate Tree Population

    Args:
        pop (Population): population to be evaluated in-place
        fitness_fn (function): fitness function that receives an individual
        pset (PrimitiveSet): set of primitives
        **kargs: keyword arguments that fitness_fn might have

    Returns:
        evaluated population by calling the fitness function
    '''
    for i in range(pop.size):
        ind = pop.individuals[i]
        ind.fitness = Fitness(fitness_fn(ind.genome, **kargs))
        depth, nodes = pg.count_tree_internals(pset, ind.genome)
        ind.depth = depth
        ind.nodes = nodes

    return pop
