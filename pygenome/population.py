import numpy as np
import pygenome.individual as individual


class Population(object):
    '''
    Base class for a population
    '''

    def __init__(self, individuals=None):
        self.individuals = individuals
        self.size = individuals.size if individual is not None else 0
    

def make_generic_population(size, make_individual_fn, *args, **kargs):
    '''
    Make Generic Population (random)

    Args:
        size (int): number of individuals in the population
        make_individual_fn (function): function that returns a random individual

    Return:
        random population of fixed size without fitness evaluation 
    '''
    pop = np.empty(size, dtype=individual.Individual)

    for i in range(size):
        pop[i] = individual.Individual()
        pop[i].genome = make_individual_fn(*args, **kargs)
       
    return Population(pop)


def make_integer_population(size, ind_size, min_value=0, max_value=1):
    '''
    Make Integer Population

    Args:
        size (int): number of individuals in the Population
        ind_size (int): the fixed size of the chromossome
        min_value (int): minimum value that can be in the chromossome
        max_value (int): maximum value (inclusive) that can be in the chromossome

    Returns:
        array of individuals randomly initialized to have integer chromossomes
    '''
    return make_generic_population(size, individual.integer_chromossome, ind_size, min_value=min_value, max_value=max_value)


def make_permutation_population(size, ind_size):
    '''
    Make Permutation Population

    Args:
        size (int): number of individuals in the Population
        ind_size (int): the fixed size of the chromossome, defines the permutation range

    Returns:
        array of individuals randomly initialized to have permutation chromossomes
    '''
    return make_generic_population(size, individual.permutation_chromossome, ind_size)


def make_uniform_population(size, ind_size, min_value=0.0, max_value=1.0):
    '''
    Make Uniform Population

    Args:
        size (int): number of individuals in the Population
        ind_size (int): the fixed size of the chromossome, defines the permutation range
        min_value (float): minimum value that can be in the chromossome
        max_value (float): maximum value (exclusive) that can be in the chromossome

    Returns:
        array of individuals randomly initialized with a uniform distribution
    '''
    return make_generic_population(size, individual.uniform_chromossome, ind_size, min_value=min_value, max_value=max_value)


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
    return make_generic_population(size, individual.normal_chromossome, ind_size, mean=mean, sigma=sigma)  