import numpy as np
import pygenome.individual as individual
import pygenome.population as population


def apply_crossover(pop, rate, cx_op, **kargs):
    '''
    Apply Crossover

    Args:
        pop (Population): individuals to apply crossover, 2 by 2
        rate (float): crossover rate
        cx_op (function): crossover operator for 2 genomes

    Return:
        population of individuals after crossover being applied in-place
    '''
    for i in range(0, pop.size - 1, step=2):
        if np.ranom.uniform() < rate:
            o1, o2 = cx_op(pop.individuals[i].genome, pop.individuals[i+1].genome, **kargs)
            pop.individuals[i].genome = o1
            pop.individuals[i+1].genome = o2
    
    return pop


def one_point_crossover(g1, g2):
    '''
    One Point Crossover

    Args:
        g1 (array): indivdiual 1 fixed genome
        g2 (array): indivdiual 2 fixed genome
    
    Returns:
        tuple with resulting offsprings genomes
    '''
    cut_point = np.random.randint(g1.size)
    o1 = np.concatenate(g1[0:cut_point], g2[cut_point:])
    o2 = np.concatenate(g2[0:cut_point], g1[cut_point:])
    
    return (o1, o2)


def uniform_crossover(g1, g2, rate=0.5):
    '''
    Uniform Crossover

    Args:
        g1 (array): indivdiual 1 fixed genome
        g2 (array): indivdiual 2 fixed genome
    
    Returns:
        tuple with resulting offsprings genomes
    '''

    o1 = np.empty(g1.size, dtype=g1.dtype)
    o1 = np.empty(g2.size, dtype=g2.dtype)
    
    for i in range(0, g1.size):
        if np.random.uniform() < rate:
            o1[i] = g1[i]
            o2[i] = g2[i]
        else:
            o1[i] = g2[i]
            o2[i] = g1[i]

    return (o1, o2)
    