import numpy as np
from copy import deepcopy

import pygenome as pg
from pygenome.representations.tree import grow_tree


class Population(object):
    '''
    Base class for a population
    '''

    def __init__(self, individuals=None):
        self.individuals = individuals
        self.size = individuals.size if individuals is not None else 0

    def clone(self):
        return deepcopy(self)


def make_empty_population(size):
    '''
    Make Empty Population

    Args:
        size (int): number of individuals that the population will contain

    Returns:
        empty population of fixed size to store new individuals
    '''
    return Population(individuals=np.empty(size, dtype=pg.Individual))


def make_generic_population(size, make_individual_fn, *args, **kargs):
    '''
    Make Generic Population

    Args:
        size (int): number of individuals in the population
        make_individual_fn (function): function that returns a random individual

    Return:
        random population of fixed size without fitness evaluation 
    '''
    pop = make_empty_population(size)

    for i in range(size):
        pop.individuals[i] = pg.Individual()
        pop.individuals[i].genome = make_individual_fn(*args, **kargs)
       
    return pop


def make_integer_population(size, ind_size, low=0, high=1):
    '''
    Make Integer Population

    Args:
        size (int): number of individuals in the Population
        ind_size (int): the fixed size of the chromossome
        low (int): minimum value that can be in the chromossome
        high (int): maximum value (inclusive) that can be in the chromossome

    Returns:
        array of individuals randomly initialized to have integer chromossomes. 
        if low and high are None, it generates a permutation from 0 to ind_size - 1
    '''
    if low is None and high is None:
        individual_type = pg.permutation_chromossome
        pop = make_generic_population(size, individual_type, ind_size)
    else: 
        individual_type = pg.integer_chromossome
        pop = make_generic_population(size, individual_type, ind_size, low=low, high=high)

    return pop


def make_uniform_population(size, ind_size, low=0.0, high=1.0):
    '''
    Make Uniform Population

    Args:
        size (int): number of individuals in the Population
        ind_size (int): the fixed size of the chromossome, defines the permutation range
        low (float): minimum value that can be in the chromossome
        high (float): maximum value (exclusive) that can be in the chromossome

    Returns:
        array of individuals randomly initialized with a uniform distribution
    '''
    return make_generic_population(size, pg.uniform_chromossome, ind_size, low=low, high=high)


def make_normal_population(size, ind_size, mean=0.0, sigma=1.0):
    '''
    Make Normal Population

    Args:
        size (int): number of individuals in the Population
        ind_size (int): the fixed size of the chromossome, defines the permutation range
        mean (float): mean of the normal distribution
        sigma (float): sigma of the normal distribution

    Returns:
        array of individuals randomly initialized with a normal distribution
    '''
    return make_generic_population(size, pg.normal_chromossome, ind_size, mean=mean, sigma=sigma)  


def make_tree_population(size, pset, max_depth, max_size, initial_type=None, init_method=grow_tree):
    '''
    Make Tree Population

    Args:
        size (int): number of individuals in the Population
        pset (PrimitiveSet): set of primitives to build a random tree
        max_depth (int): initial max tree depth
        max_size (int): max tree size that translates into max array size
        initial_type (type): when using types, this constraints the initial primitive ot be of this type
        init_method (function): function that generates random trees (grow_tree, full_tree)  
    
    Returns:
        array of tree based individuals initialized according to given method, with or without types
    '''
    pop = make_empty_population(size)

    for i in range(size):
        pop.individuals[i] = pg.TreeIndividual()
        pop.individuals[i].genome = init_method(pset, max_depth, max_size, initial_type=initial_type)
        pop.individuals[i].depth = pg.count_tree_depth(pop.individuals[i].genome)
        pop.individuals[i].nodes = pg.count_tree_nodes(pop.individuals[i].genome)
               
    return pop
