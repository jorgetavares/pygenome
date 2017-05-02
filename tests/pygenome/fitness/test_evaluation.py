import numpy as np
import pygenome as pg
import operator as op

def test_Fitness():
    fit = pg.Fitness()
    assert fit.value is None
    fit = pg.Fitness(value=1.0)
    assert fit.value == 1.0

def test_evaluate_population():
    np.random.seed(42)
    size = 1000
    ind_size = 100
    fitness_function = lambda x : 1. / pg.onemax(x) 
    pop = pg.make_integer_population(size, ind_size)
    original_pop = pop.clone()
    pop = pg.evaluate_population(pop, fitness_function)
    for i in range(pop.size):
        assert pop.individuals[i].run_eval is False
        assert pop.individuals[i].run_eval is not original_pop.individuals[i].run_eval
        assert type(pop.individuals[i].fitness) is pg.Fitness
        assert pop.individuals[i].fitness.value is not None

def protected_div(a, b):
    if b > 0:
        return op.truediv(a, b)
    else:
        return b

def setup_primitive_set():
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

    return pset

# define problem
# y = f(x) = x^2 / 2
def fn(x):
    return protected_div(op.mul(x, x), 2.0)

def test_evaluate_tree_population():
    np.random.seed(42)
    size = 50
    rate = 1.0
    initial_max_depth = 6
    max_depth = 12
    pset = setup_primitive_set()
    num_fitness_cases = 10
    fitness_function = pg.make_fitness_regression(pset, fn, num_fitness_cases)
    pop = pg.make_tree_population(size, pset, initial_max_depth, max_depth, init_method=pg.full_tree)
    original_pop = pop.clone()
    pop = pg.evaluate_tree_population(pop, fitness_function, pset)
    for i in range(pop.size):
        assert pop.individuals[i].run_eval is False
        assert pop.individuals[i].run_eval is not original_pop.individuals[i].run_eval
        assert type(pop.individuals[i].fitness) is pg.Fitness
        assert pop.individuals[i].fitness.value is not None
        assert pop.individuals[i].depth == original_pop .individuals[i].depth
        assert pop.individuals[i].nodes == original_pop .individuals[i].nodes
