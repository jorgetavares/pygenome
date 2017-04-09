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

pset = pg.PrimitiveSet()
# functions
pset.addFunction(op.add, 2)
pset.addFunction(op.sub, 2)
pset.addFunction(op.mul, 2)
pset.addFunction(protected_div, 2)
# constants
num_constants = 10
for i in range(num_constants):
    pset.addTerminal(np.random.randint(-5, 5))
# variables
pset.addVariable("x")


# define problem
# y = f(x) = x^2 / 2
def fn(x):
    return protected_div(op.mul(x, x), 2.0)

num_fitness_cases = 10
fitness_fn = make_fitness_regression(pset, fn, num_fitness_cases)


# GP
def gp_with_elitism():
    np.random.seed(42)
    pop = pg.genetic_programming(fitness_fn, pset, min_depth, max_depth. max_size, elitism=True, total_generations=20)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# entry point
if __name__ == "__main__":
    print('GP: with elitism')
    gp_with_elitism()    
 