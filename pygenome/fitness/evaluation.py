import numpy as np
import pygenome as pg


class Fitness(object):
    '''
    Base class for a Fitness
    '''

    def __init__(self, value=None):
        self.value = value


def evaluate_population(pop, fitness_fn, **kargs):
    '''
    Evaluate Population

    Args:
        pop (Population): population to be evaluated in-place
        fitness_fn (function): fitness function that receives an individual
        **kargs: keyword arguments that fitness_fn might have

    Returns:
        evaluated population by calling the fitness function
    '''
    for i in range(pop.size):
        ind = pop.individuals[i]
        ind.fitness = Fitness(fitness_fn(ind.genome, **kargs))
    
    return pop


def evaluate_tree_population(pop, fitness_fn, pset, **kargs):
    '''
    Evaluate Tree Population

    Args:
        pop (Population): population to be evaluated in-place
        fitness_fn (function): fitness function that receives an individual
        pset (PrimitiveSet): set of primitives
        **kargs: keyword arguments that fitness_fn might have

    Returns:
        evaluated population by calling the fitness function
    '''
    for i in range(pop.size):
        ind = pop.individuals[i]
        ind.fitness = Fitness(fitness_fn(ind.genome, **kargs))
        depth, nodes = pg.count_tree_internals(pset, ind.genome)
        ind.depth = depth
        ind.nodes = nodes

    return pop


def make_fitness_regression(pset, fn, num_fitness_cases, loss=pg.mean_squared_error):
    
    x_points = np.asarray([x for x in range(num_fitness_cases)])
    y_points = np.asarray([fn(x) for x in x_points])
    
    def regression(solution):
        vars_inputs = {}
        x_evals = np.empty(num_fitness_cases)
        for i in range(num_fitness_cases)
            vars_inputs["x"] = x_points[i]
            x_evals[i] = pg.interpreter(pset, solution, run=True, vars_inputs=vars_inputs)
        
        return loss(x_evals, y_points)

    return regression
