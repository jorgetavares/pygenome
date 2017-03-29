import numpy as np
import pygenome as pg


def generational_replacement(pop, offsprings):
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


def steady_state_replacement(pop, offsprings):
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


# ES stype replacement

def mu_comma_lambda_replacement(pop, offsprings):
    '''
    Mu Comma Lambda replacement (mu, lambda)

    Args:
        pop (Population): original individuals
        offsprings (Popultion): offpsrings that will be inserted into pop

    Returns:
        popuplation from sorted offsprings
    '''
    # get offpsrings sorted indexes
    fitness_values = [i.fitness.value for i in offsprings.individuals]
    sort_indexes = np.argsort(fitness_values)
    
    # replace population with sorted offpsrings
    for i in range(pop.size):
        pop.individuals[i] = offsprings.individuals[sort_indexes[i]].clone()

    return pop 


def mu_plus_lambda_replacement(pop, offsprings):
    '''
    Mu Plus Lambda replacement (mu + lambda)

    Args:
        pop (Population): original individuals
        offsprings (Popultion): offpsrings that will be inserted into pop

    Returns:
        popuplation from sorted original population plus offsprings
    '''
    # joins individuals and get sorted indexes 
    joint_pop = np.concatenate((pop.individuals, offsprings.individuals))
    fitness_values = [i.fitness.value for i in joint_pop]
    sort_indexes = np.argsort(fitness_values)

    # replace population with sorted joint populaton
    for i in range(pop.size):
        pop.individuals[i] = joint_pop[sort_indexes[i]].clone()

    return pop
