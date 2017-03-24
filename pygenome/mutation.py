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


def flip_mutation(chromossome, gene_rate=None, low=0, high=1, **kargs):
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


def swap_mutation(chromossome, gene_rate=None, **kargs):
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


def uniform_mutation(chromossome, gene_rate=None, **kargs):
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


# ES functions
def apply_global_mutation(pop, pool_size, mt_op, **kargs):
    '''
    Apply Global Mutation

    Args:
        pop (Population): individuals to apply mutation, randomly picked until pool_size
        pool_size (int): number of individuals that will be applied mutation
        mt_op (function): mutation operator
        **kargs: keyword arguments for the crossover oeprator

    Return:
        population of mutated individuals 
    '''
    new_pop = population.make_empty_population(pool_size)

    for i in range(0, pool_size):
        new_pop.individuals[i].genome = mt_op(pop.individuals[np.random.randint(pop.size)].genome, **kargs)
    
    return new_pop


def sigma_check(value, epsilon):
    return epsilon if value < epsilon else value


def uncorrelated_one_step_mutation(chromossome, epsilon):
    tau = 1.0 / np.sqrt(chromossome.size - 1)
    sigma = sigma_check(chromossome[-1] * np.exp(tau * np.random.normal(), epsilon))

    offspring = np.empty(chromossome.size, dtype=chromossome.dtype=float)

    for i in range(chromossome.size):
        offspring[i] = chromossome[i] + sigma * np.random.normal()

    offspring[-1] = sigma

    return offspring


def uncorrelated_n_steps_mutation(chromossome, epsilon):
    n = chromossome.size / 2
    tau1 = 1.0 / np.srqt(2.0 * n)
    tau2 = 1.0 / np.srqt(2.0 * np.sqrt(n))
    
    sigmas = np.array(chromossome[n:])
    for i in range(sigmas.size):
        sigmas[i] = sigma_check(sigmas[i] * np.exp(tau1 * np.random.normal() + tau2 * np.random.normal()), epsilon)

    values = np.array(chromossome[:n])
    for i in range(values.size):
        values[i] = values[i] + sigmas[i] * np.random.normal()

    return np.concatenate((values, sigmas))
    