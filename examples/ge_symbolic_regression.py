import sys
sys.path.append('../')

import numpy as np
import pygenome as pg

import operator as op


def protected_div(a, b):
    if b > 0:
        return op.truediv(a, b)
    else:
        return b

# define problem
# y = f(x) = x^2 / 2


def fn(x):
    return protected_div(op.mul(x, x), 2.0)


def ge_with_elitism():
    np.random.seed(42)

    ind_size = 100
    low = 0
    high = 255
    wrap = True

    grammar = Grammar(filename='ge_symbolic_regression_grammar.txt')
    num_fitness_cases = 10
    fitness_fn =  # TODO

    pop = pg.grammatical_evolution(
        grammar, fitness_fn,  ind_size, low, high, elitism=True, total_generations=20, pop_size=100)

    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))


# entry point
if __name__ == "__main__":
    print('GE: with elitism')
    ge_with_elitism()
