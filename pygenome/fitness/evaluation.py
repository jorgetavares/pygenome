import numpy as np
import pygenome as pg

from pygenome.fitness.loss import mean_squared_error


class Fitness(object):
    '''
    Base class for a Fitness
    '''

    def __init__(self, value=None):
        self.value = value

    def equal(self, fit):
        if fit is None:
            return False
        else:
            return self.value == fit.value


def evaluate_population(pop, fitness_fn, map_fn=None, **kargs):
    '''
    Evaluate Population

    Args:
        pop (Population): population to be evaluated in-place
        fitness_fn (function): fitness function that receives an individual
        map_fn (function): mapping function from genotype to phenotype
        **kargs: keyword arguments that fitness_fn might have

    Returns:
        evaluated population by calling the fitness function
    '''
    for i in range(pop.size):
        ind = pop.individuals[i]
        if ind.run_eval:
            if map_fn is None:
                ind.fitness = Fitness(fitness_fn(ind.genotype, **kargs))
            else:
                ind.fitness = Fitness(fitness_fn(
                    map_fn(ind.genotype), **kargs))
            ind.run_eval = False

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
        if ind.run_eval:
            ind.fitness = Fitness(fitness_fn(ind.genotype, **kargs))
            depth, nodes = pg.count_tree_internals(pset, ind.genotype)
            ind.depth = depth
            ind.nodes = nodes
            ind.run_eval = False

    return pop


# TODO: this should be refactored and generalized to allow better definition
# of fitness fucntions for (ST)GP. Right now, it acts more of an example of
# how to build an evaluator for the data/problem
def make_fitness_regression(pset, fn, num_fitness_cases, loss=mean_squared_error):
    '''
    Make Fitness Regression (1 variable)

    Args:
        pset (PrimitiveSet): set of functions, terminals and variables
        fn (functions): objective functions
        num_fitness_cases (int): number of features
        loss (functions): loss function

    Returns:
        function that receives a GP solution and evaluates it for regression (1 variable)
    '''
    x_points = np.asarray([x for x in range(num_fitness_cases)])
    y_points = np.asarray([fn(x) for x in x_points])
    variable, _ = (list(pset.variables.values()))[0]

    def regression(solution):
        vars_inputs = {}
        x_evals = np.empty(num_fitness_cases)
        for i in range(num_fitness_cases):
            vars_inputs[variable] = x_points[i]
            x_evals[i] = pg.interpreter(
                pset, solution, run=True, vars_inputs=vars_inputs)

        return loss(x_evals, y_points)

    return regression



def evaluate_grammar_population(pop, grammar, wrap, fitness_fn, **kargs):
    '''
    Evaluate Population

    Args:
        pop (Population): population to be evaluated in-place
        grammar (Grammar): gramar to convert a genotype into a phenotype
        fitness_fn (function): fitness function that receives an individual
        **kargs: keyword arguments that fitness_fn might have

    Returns:
        evaluated population by calling the fitness function
    '''
    for i in range(pop.size):
        ind = pop.individuals[i]
            ind.fitness = Fitness(fitness_fn(grammar.mapDerivative(
                grammar.start_symbol, ind.genotype, wrap=wrap), **kargs))
            ind.fitness = Fitness(fitness_fn(grammar.mapDerivative(

            ind.run_eval = False

    return pop
