import numpy as np
import pygenome.individual as individual


class Population(object):
    '''
    Base class for a population
    '''

    def __init__(self):
        self.individuals = None


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
       
    return pop


def make_integer_population(size, ind_size, min_value=0, max_value=1):
    '''
    Make Integer Population

    Args:
        size (int): number of individuals in the Population
        ind_size (int): the fixed size of the chromossome
        min_value (int): minimum value that can be in the chromossome
        max_value (int): maximum value (exclusive) that can be in the chromossome

    Returns:
        array of individuals randomly initialized to have integer chromossomes
    '''
    return make_generic_population(size, individual.integer_chromossome, ind_size, min_value=min_value, max_value=max_value)