import numpy as np
import pygenome.individual as individual


class Population(object):
    '''
    Base class for a population
    '''

    def __init__(self, individuals=None):
        self.individuals = individuals
        self.size = individuals.size if individual is not None else 0

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
    return Population(individuals=np.empty(size, dtype=individual.Individual))


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
        pop.individuals[i] = individual.Individual()
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
        individual_type = individual.permutation_chromossome
        pop = make_generic_population(size, individual_type, ind_size)
    else: 
        individual_type = individual.integer_chromossome
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
    return make_generic_population(size, individual.uniform_chromossome, ind_size, low=low, high=high)


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