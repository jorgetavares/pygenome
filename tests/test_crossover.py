import sys
sys.path.append('../')

import numpy as np
import pygenome as pg


def test_apply_crossover1():
    np.random.seed(42)
    size = 1000
    ind_size = 100
    rate = 1.0
    operator = pg.uniform_crossover
    pop = pg.make_integer_population(size, ind_size)
    original_pop = pop.clone()
    pop = pg.apply_crossover(pop, rate, operator)
    for i in range(pop.size):
        assert pop.individuals[i].run_eval == True
        assert np.array_equal(pop.individuals[i].genotype, original_pop.individuals[i].genotype) is not True

def test_apply_crossover2():
    np.random.seed(42)
    size = 1000
    ind_size = 100
    rate = 0.0
    operator = pg.uniform_crossover
    pop = pg.make_integer_population(size, ind_size)
    original_pop = pop.clone()
    pop = pg.apply_crossover(pop, rate, operator)
    for i in range(pop.size):
        assert pop.individuals[i].run_eval == True
        assert np.array_equal(pop.individuals[i].genotype, original_pop.individuals[i].genotype)

def test_one_point_crossover():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]))
    i2 = pg.Individual(genotype=np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1]))
    o1, o2 = pg.one_point_crossover(i1, i2)
    assert np.array_equal(o1.genotype, np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1])) 
    assert np.array_equal(o2.genotype, np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1])) 

def test_uniform_crossover():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]))
    i2 = pg.Individual(genotype=np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1]))
    o1, o2 = pg.uniform_crossover(i1, i2)
    assert np.array_equal(o1.genotype, np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1])) 
    assert np.array_equal(o2.genotype, np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1])) 

def test_partially_match_crossover():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6]))
    i2 = pg.Individual(genotype=np.array([8, 9, 1, 3, 6, 5, 7, 4, 2, 0]))
    o1, o2 = pg.partially_match_crossover(i1, i2)
    assert np.array_equal(o1.genotype, np.array([4, 7, 8, 3, 6, 5, 1, 2, 0, 9])) 
    assert np.array_equal(o2.genotype, np.array([9, 3, 6, 7, 8, 1, 5, 4, 2, 0])) 

def test_apply_global_crossover():
    np.random.seed(42)
    size = 1000
    ind_size = 100
    operator = pg.intermediary_crossover
    pop = pg.make_uniform_population(size, ind_size)
    original_pop = pop.clone()
    pop = pg.apply_global_crossover(pop, operator)
    for i in range(pop.size):
        assert pop.individuals[i].run_eval == True
