import numpy as np
import pygenome as pg


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


def apply_tree_mutation(pop, rate, mt_op, **kargs):
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
            offspring = mt_op(pop.individuals[i].genome, pop.individuals[i].depth, pop.individuals[i].nodes, **kargs)
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


def uniform_mutation(chromossome, gene_rate=None, low=0.0, high=1.0, **kargs):
    '''
    Uniform Mutation

    Args:
        chromossome (array): float chromossome to be mutated
        rate (float): per gene mutation rate

    Returns:
        mutated chromossome
    '''
    rate = 1. / chromossome.size if gene_rate is None else gene_rate
    for i in range(0, chromossome.size):
        if np.random.uniform() < rate:
            chromossome[i] = np.random.uniform(low=low, high=high)

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
        index = np.random.randint(pop.size)
        new_pop.individuals[i] = individual.Individual()
        new_pop.individuals[i].genome = mt_op(pop.individuals[index].genome, **kargs)
    
    return new_pop


def sigma_check(sigma, epsilon=1e-08):
    '''
    Sigma Check

    Args:
        sigma (float): sigma value to be bounded
        epsilon (flota): minimum vale for sigma
    
    Returns:
        bounded sigma
    '''
    return epsilon if sigma < epsilon else sigma


def uncorrelated_one_step_mutation(chromossome, epsilon=1e-08):
    '''
    Uncorrelated One Step Mutation

    Args:
        chromossome (float array): chromossome to be mutated. Last gene is the sigma value
        epsilon (float): minimum value sigma can have

    Returns:
        mutated chromossome
    '''
    tau = 1.0 / np.sqrt(chromossome.size - 1)
    sigma = sigma_check(chromossome[-1] * np.exp(tau * np.random.normal()), epsilon)

    offspring = np.empty(chromossome.size, dtype=chromossome.dtype)

    for i in range(chromossome.size):
        offspring[i] = chromossome[i] + sigma * np.random.normal()

    offspring[-1] = sigma # last is the self-adaptive rate, not part of the solution

    return offspring


def uncorrelated_n_steps_mutation(chromossome, epsilon=1e-08):
    '''
    Uncorrelated N Steps Mutation

    Args:
        chromossome (float array): chromossome to be mutated. 
        First half is the problem, second half sigma values
        epsilon (float): minimum value sigma can have

    Returns:
        mutated chromossome
    '''
    n = int(chromossome.size / 2)
    tau1 = (1.0 / np.sqrt(2.0 * n)) * np.random.normal()
    tau2 = 1.0 / np.sqrt(2.0 * np.sqrt(n))
    
    sigmas = np.array(chromossome[n:])
    for i in range(sigmas.size):
        sigmas[i] = sigma_check(sigmas[i] * np.exp(tau1 + tau2 * np.random.normal()), epsilon)

    values = np.array(chromossome[:n])
    for i in range(values.size):
        values[i] = values[i] + sigmas[i] * np.random.normal()

    return np.concatenate((values, sigmas))
    

def subtree_mutation(t1, t1_depth, t1_nodes, pset=None, **kargs):
    return np.copy(t1)
