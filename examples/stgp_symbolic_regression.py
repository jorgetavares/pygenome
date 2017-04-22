import sys
sys.path.append('../')

import numpy as np
import pygenome as pg


# define Primitive Set 
import operator as op

def protected_div(a, b):
    if b > 0:
        return op.truediv(a, b)
    else:
        return b
    
def float_add(a, b):
    return op.add(a, b)

def float_sub(a, b):
    return op.sub(a, b)


def setup_primitive_set():
    pset = pg.PrimitiveSet(typed=True)
    # functions
    pset.addFunction(op.add, 2, [int, float, float])
    pset.addFunction(op.sub, 2, [int, float, float])
    pset.addFunction(float_add, 2, [float, int, int])
    pset.addFunction(float_sub, 2, [float, int, int])
    pset.addFunction(op.mul, 2, [int, int, int]) 
    pset.addFunction(protected_div, 2, [float, float, float])
    # constants
    num_constants = 5
    for i in range(num_constants):
        pset.addTerminal(np.random.randint(-5, 5), [int])
    for i in range(num_constants):
        pset.addTerminal(np.random.uniform(), [float])
    # variables
    pset.addVariable("x", [float])

    return pset

# define problem
# y = f(x) = x^2 / 2
def fn(x):
    return protected_div(op.mul(x, x), 2.0)

# GP
def stgp_with_elitism():
    np.random.seed(42)
    
    init_min_depth = 2
    init_max_depth = 6
    max_tree_depth = 12
    
    pset = setup_primitive_set()
    
    num_fitness_cases = 10
    fitness_fn = pg.make_fitness_regression(pset, fn, num_fitness_cases)
    
    pop = pg.genetic_programming(fitness_fn, pset, init_min_depth, init_max_depth, max_tree_depth, elitism=True, total_generations=100, pop_size=1000, cx_rate=0.9, op_mt_rate=0.25)
    
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genotype))

# entry point
if __name__ == "__main__":
    print('STGP: with elitism')
    stgp_with_elitism()    
 