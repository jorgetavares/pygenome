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
    new_pop = population.make_empty_population(pop.size)

    for i in range(new_pop.size):
        new_pop.individuals[i] = selection_fn(pop, **kargs)
        
    return new_pop


def select_steadystate_population(pop, selection_fn, steady_state_size=2, **kargs):
    '''
    Select Steady-State Population

    Args:
        pop (Population): population to apply selection
        selection_fn (function): selection function
        steady_state_size (int): size of parents to select
        **kargs: keyword arguments that selection_fn might have

    Returns:
        selected  individuals for steady state mating 
    '''
    new_pop = population.make_empty_population(steady_state_size)

    for i in range(steady_state_size):
        new_pop.individuals[i] = selection_fn(pop, **kargs)
        
    return new_pop


def best_individual(pop):
    '''
    Best individual

    Args:
        pop (Population): population of individuals

    Returns:
        return a clone of the best individual in the population
    '''
    best_idx = 0
    best_fit = pop.individuals[best_idx].fitness.value

    for i in range(1, pop.size):
        if pop.individuals[i].fitness.value < pop.individuals[best_idx].fitness.value:
            best_idx = i
            best_fit = pop.individuals[i].fitness.value
    
    return pop.individuals[best_idx].clone()


def tournament_selection(pop, size=3):
    '''
    Tournament Selection (minimization)

    Args:
        pop (Population): population to select from
        size (int): the tournament size
    
    Returns:
        cloned selected individual
    '''
    best = pop.individuals[np.random.randint(pop.size)]

    for step in range(2, size):
        current = pop.individuals[np.random.randint(pop.size)]
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
    