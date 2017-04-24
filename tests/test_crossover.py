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

def test_intermediary_crossover():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998, 0.78517596,  0.19967378,  0.51423444,  0.59241457,  0.04645041]))
    i2 = pg.Individual(genotype=np.array([ 0.60754485,  0.17052412,  0.06505159,  0.94888554,  0.96563203, 0.80839735,  0.30461377,  0.09767211,  0.68423303,  0.44015249]))
    o1 = pg.intermediary_crossover(i1, i2)
    assert np.isclose(o1.genotype.all(), np.array([ 0.60969887,  0.15500899,  0.17859812,  0.65762369,  0.710851  ,
        0.79678666,  0.25214378,  0.30595327,  0.6383238 ,  0.24330145]).all()) 
    
def test_disrete_crossover():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998,
        0.78517596,  0.19967378,  0.51423444,  0.59241457,  0.04645041]))
    i2 = pg.Individual(genotype=np.array([ 0.60754485,  0.17052412,  0.06505159,  0.94888554,  0.96563203,
        0.80839735,  0.30461377,  0.09767211,  0.68423303,  0.44015249]))
    o1 = pg.discrete_crossover(i1, i2)
    assert np.isclose(o1.genotype.all(), np.array([ 0.61185289,  0.17052412,  0.06505159,  0.94888554,  0.45606998,
        0.78517596,  0.19967378,  0.09767211,  0.68423303,  0.44015249]).all()) 
   