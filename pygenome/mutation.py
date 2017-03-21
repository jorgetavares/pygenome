import numpy as np
import pygenome.individual as individual
import pygenome.population as population


def apply_mutation(pop, rate, mt_op, **kargs):
    '''
    Apply Mutation

    Args:
        pop (Population): individuals to apply mutation
        rate (float): mutation rate
        mt_op (function): mutation operator for 1 genome

    Return:
        population of individuals after mutation being applied in-place
    '''
    for i in range(0, pop.size):
        if np.random.uniform() < rate:
            offspring = mt_op(pop.individuals[i].genome, **kargs)
            pop.individuals[i].genome = offspring
    
    return pop


def flip_mutation(chromossome, gene_rate=None, low=0, high=1):
    '''
    Flip Mutation

    Args:
        chromossome (array): integer chromossome to be mutated
        rate (float): per gene mutation rate
        low (int): low value the chromossome can have
        hight (int): high value (inclusive) the chromossome can have

    Returns:
        mutated chromossome
    '''
    rate = 1. / chromossome.size if gene_rate is None else gene_rate
    for i in range(0, chromossome.size):
        high_value = high + 1 # to be inclusive
        if np.random.uniform() < rate:
            chromossome[i] = np.random.randint(low, high=high_value)

    return chromossome


def swap_mutation(chromossome, gene_rate=None):
    '''
    Swap Mutation

    Args:
        chromossome (array): integer chromossome to be mutated
        rate (float): per gene mutation rate

    Returns:
        mutated chromossome where each gene can be swapped with 
        another one so that permutations can be preserved 
    '''
    rate = 1. / chromossome.size if gene_rate is None else gene_rate
    for i in range(0, chromossome.size):
        if np.random.uniform() < rate:
            j = np.random.randint(chromossome.size)
            temp = chromossome[i]
            chromossome[i] = chromossome[j]
            chromossome[j] = temp

    return chromossome


def uniform_mutation(chromossome, gene_rate=None):
    '''
    Uniform Mutation

    Args:
        chromossome (array): float chromossome to be mutated
        rate (float): per gene mutation rate

    Returns:
        mutated chromossome
    '''
    for i in range(0, chromossome.size):
        if np.random.uniform() < rate:
            chromossome[i] = np.random.uniform()

    return chromossome