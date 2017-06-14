import numpy as np
import pygenome as pg


# define Primitive Set
import operator as op


def protected_div(a, b):
    if b > 0:
        return op.truediv(a, b)
    else:
        return b


def random_ints():
    return np.random.randint(-5, 5)


def setup_primitive_set():
    pset = pg.PrimitiveSet()
    # functions
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addFunction(op.mul, 2)
    pset.addFunction(protected_div, 2)
    # constants
    pset.addTerminal(random_ints, ephemeral=True)
    # variables
    pset.addVariable("x")

    return pset

# define problem
# y = f(x) = x^2 / 2


def fn(x):
    return protected_div(op.mul(x, x), 2.0)

# GP


def gp_with_elitism_ephemeral():
    np.random.seed(42)

    init_max_depth = 4
    max_tree_depth = 8

    pset = setup_primitive_set()

    num_fitness_cases = 10
    fitness_fn = pg.make_fitness_regression(pset, fn, num_fitness_cases)

    pop = pg.genetic_programming(fitness_fn, pset, init_max_depth,
                                 max_tree_depth, elitism=True, total_generations=20, pop_size=1000)

    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))


# entry point
if __name__ == "__main__":
    print('GP: with elitism using ephemeral constants')
    gp_with_elitism_ephemeral()
