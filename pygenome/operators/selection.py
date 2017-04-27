import numpy as np
import pygenome as pg


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
    new_pop = pg.make_empty_population(pop.size)

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
    new_pop = pg.make_empty_population(steady_state_size)

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


def worst_individual(pop):
    '''
    Worst individual

    Args:
        pop (Population): population of individuals

    Returns:
        return a clone of the worst individual in the population
    '''
    worst_idx = 0
    worst_fit = pop.individuals[worst_idx].fitness.value

    for i in range(1, pop.size):
        if pop.individuals[i].fitness.value > pop.individuals[worst_idx].fitness.value:
            worst_idx = i
            worst_fit = pop.individuals[i].fitness.value
    
    return pop.individuals[worst_idx].clone()


def random_selection(pop):
    '''
    Random Selection

    Args:
        pop (population): population of individuals

    Returns:
        random cloned individual 
    '''
    return pop.individuals[np.random.randint(pop.size)].clone()


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


def negative_tournament_selection(pop, size=3):
    '''
    Negative Tournament Selection (minimization)

    Args:
        pop (Population): population to select from
        size (int): the tournament size
    
    Returns:
        cloned selected individual
    '''
    worst = pop.individuals[np.random.randint(pop.size)]

    for step in range(2, size):
        current = pop.individuals[np.random.randint(pop.size)]
        worst = current if current.fitness.value > worst.fitness.value else worst

    return worst.clone()


def roulette_wheel_selection(pop):
    '''
    Roulette Wheel Selection

    Args:
        pop (Population): population to select from
    
    Returns:
        cloned selected individual
    '''
    from bisect import bisect_left

    # this is required since this selection method 
    # only works in maximization setup
    def invert_fitness(value):
        return 1.0 / (1.0 + value)

    fitness_values = [invert_fitness(pop.individuals[i].fitness.value) for i in range(pop.size)]
    cfs = [sum(fitness_values[:i+1]) for i in range(pop.size)]
    i = bisect_left(cfs, np.random.uniform(low=0.0, high=cfs[-1]))

    return pop.individuals[i].clone()
    