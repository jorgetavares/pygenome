import sys
sys.path.append('../')

import numpy as np
import pygenome as pg

def test_make_population():
    pop = pg.Population()
    assert type(pop) is pg.Population
    assert pop.individuals is None
    assert pop.size == 0

def test_clone_population():
    # for this test we don't require a real population of Individual objects
    inds = np.array([1, 2, 3, 4, 5])
    pop1 = pg.Population(individuals=inds)
    pop2 = pop1.clone()
    assert np.array_equal(pop1.individuals, pop2.individuals)
    assert pop1.size == pop2.size

    pop1.individuals = np.array([6, 7, 8])
    pop1.size = 3
    assert np.array_equal(pop1.individuals, pop2.individuals) is False
    assert pop2.size == 5

def test_make_empty_population():
    size = 10
    pop = pg.make_empty_population(size)
    assert type(pop) is pg.Population
    assert pop.size == size
    assert pop.individuals.size == size
    for i in range(size):
        assert pop.individuals[i] is None

def test_make_binary_population():
    size = 10
    ind_size = 20
    pop = pg.make_integer_population(size, ind_size)
    assert type(pop) is pg.Population
    assert pop.size == size
    assert pop.individuals.size == size
    for i in range(size):
        assert type(pop.individuals[i]) is pg.Individual
        assert pop.individuals[i].genotype.size == ind_size
        for g in range(ind_size):
            assert pop.individuals[i].genotype[g] >= 0 and pop.individuals[i].genotype[g] <= 1

def test_make_integer_population():
    size = 10
    ind_size = 20
    low = 0
    high = 19
    pop = pg.make_integer_population(size, ind_size, low=low, high=high)
    assert type(pop) is pg.Population
    assert pop.size == size
    assert pop.individuals.size == size
    for i in range(size):
        assert type(pop.individuals[i]) is pg.Individual
        assert pop.individuals[i].genotype.size == ind_size
        for g in range(ind_size):
            assert pop.individuals[i].genotype[g] >= low and pop.individuals[i].genotype[g] <= high


