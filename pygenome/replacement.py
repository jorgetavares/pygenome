import numpy as np
import pygenome.population as population


def generational(pop, offsprings):
    '''
    Generational replacement

    Args:
        pop (Population): original individuals
        offsprings (Popultion): offpsrings that will be inserted into pop

    Returns:
        original pop completely replaced by the offsprings
    '''
    if pop.size == offsprings.size:
        # TODO: analyze if a deepcopy operation is required
        pop.individuals = offsprings.individuals
    else:
        raise IndexError('generational replacement: pop.size != offsprings.size')

    return pop


def steady_state(pop, offsprings):
    '''
    Steady State replacement

    Args:
        pop (Population): original individuals
        offsprings (Popultion): offpsrings that will be inserted into pop

    Returns:
        original pop with offpsrings randomly inserted, replacing some parents
    '''
    for offspring in offsprings.individuals:
        index = np.random.randint(pop.size)
        # TODO: analyze if a deepcopy operation is required
        pop.individuals[index] = offspring

    return pop 


def elite_strategy(pop, best):
    '''
    Elite Strategy replacement

    Args:
        pop (Population): original individuals
        best (Individual): individual to be randomly inserted into pop

    Returns:
        original pop with best individual randomly inserted
    '''
    # TODO: analyze if a deepcopy operation is required
    pop.individuals[np.random.randint(pop.size)] = best

    return pop