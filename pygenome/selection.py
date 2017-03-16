import numpy as np

import pygenome.individual as individual
import pygenome.population as population
import pygenome.fitness as fitness


def select_population(pop, selection_fn, **kargs):
    '''
    Select Population

    Args:
        pop (Population): population to apply selection
        selection_fn (function): selection function
        **kargs: keyword arguments that selection_fn might have

    Returns:
        selected population for mating 
    '''
    new_pop = make_empty_population(pop.size)

    for i in range(new_pop.size):
        new_pop.individuals[i] = selection_fn(pop, **kargs)
        
    return new_pop


def tournament_selection(pop, size=3):
    '''
    Tournament Selection (minimization)

    Args:
        pop (Population): population to select from
        size (int): the tournament size
    
    Returns:
        cloned selected individual
    '''
    best = population.individuals[np.random.randint(pop.size)]

    for step in range(2, size):
        current = population.individuals[np.random.randint(pop.size)]
        best = current if current.fitness.value < best.fitness.value else best

    return best.clone()


def roulette_wheel_selection(pop):
    '''
    Roulette Wheel Selection

    Args:
        pop (Population): population to select from
    
    Returns:
        cloned selected individual
    '''
    total = 0.0
    for ind in pop.individuals:
        total += ind.fitness.value
    
    probability = np.empty(pop.size)
    for i in range(1, pop.size):
        probability[i] = pop.individuals[i].fitness.value / total

    n = 0
    total_sum = probability[n]
    rand = np.random.uniform()

    while total_sum < rand:
        n += 1
        total_sum += probability[n]

    return pop.indvidiuals[n].clone()
    