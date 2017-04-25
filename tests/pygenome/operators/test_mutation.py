import sys
sys.path.append('../')

import numpy as np
import pygenome as pg


def test_apply_mutation1():
    np.random.seed(42)
    size = 1000
    ind_size = 100
    rate = 1.0
    operator = pg.flip_mutation
    pop = pg.make_integer_population(size, ind_size)
    original_pop = pop.clone()
    pop = pg.apply_mutation(pop, rate, operator, gene_rate=rate)
    for i in range(pop.size):
        assert pop.individuals[i].run_eval == True
        assert np.array_equal(pop.individuals[i].genotype, original_pop.individuals[i].genotype) is not True

def test_apply_mutation2():
    np.random.seed(42)
    size = 1000
    ind_size = 100
    rate = 0.0
    operator = pg.flip_mutation
    pop = pg.make_integer_population(size, ind_size)
    original_pop = pop.clone()
    pop = pg.apply_mutation(pop, rate, operator, gene_rate=rate)
    for i in range(pop.size):
        assert pop.individuals[i].run_eval == True
        assert np.array_equal(pop.individuals[i].genotype, original_pop.individuals[i].genotype)


