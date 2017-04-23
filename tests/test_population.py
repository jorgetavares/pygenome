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

def test_make_permutation_population():
    size = 10
    ind_size = 20
    pop = pg.make_integer_population(size, ind_size, low=None, high=None)
    assert type(pop) is pg.Population
    assert pop.size == size
    assert pop.individuals.size == size
    for i in range(size):
        assert type(pop.individuals[i]) is pg.Individual
        assert pop.individuals[i].genotype.size == ind_size
        for g in range(ind_size):
            assert pop.individuals[i].genotype[g] >= 0 and pop.individuals[i].genotype[g] < ind_size
        assert np.array_equal(np.sort(pop.individuals[i].genotype), np.array([g for g in range(ind_size)]))

def test_make_uniform_population():
    size = 10
    ind_size = 20
    low = 0.0
    high = 1.0
    pop = pg.make_uniform_population(size, ind_size, low=low, high=high)
    assert type(pop) is pg.Population
    assert pop.size == size
    assert pop.individuals.size == size
    for i in range(size):
        assert type(pop.individuals[i]) is pg.Individual
        assert pop.individuals[i].genotype.size == ind_size
        for g in range(ind_size):
            assert pop.individuals[i].genotype[g] >= low and pop.individuals[i].genotype[g] <= high

def test_make_normal_population():
    size = 10
    ind_size = 20
    mean = 0.0
    sigma = 1.0
    pop = pg.make_normal_population(size, ind_size, mean=mean, sigma=sigma)
    assert type(pop) is pg.Population
    assert pop.size == size
    assert pop.individuals.size == size
    for i in range(size):
        assert type(pop.individuals[i]) is pg.Individual
        assert pop.individuals[i].genotype.size == ind_size
        # TODO: add function to test normality of distribution?

def test_make_tree_population():
    import operator as op 
    np.random.seed(42)
    size = 10
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    initial_min_depth = 2
    initial_max_depth = 6
    max_depth = 8
    pop = pg.make_tree_population(size, pset, initial_min_depth, initial_max_depth, max_depth, init_method=pg.full_tree)
 
    assert type(pop) is pg.Population
    assert pop.size == size
    assert pop.individuals.size == size
    for i in range(size):
        assert type(pop.individuals[i]) is pg.TreeIndividual
        assert pop.individuals[i].depth >= initial_min_depth and pop.individuals[i].depth <= initial_max_depth
        pos = 0
        while pos < pop.individuals[i].genotype.size and pop.individuals[i].genotype[pos] != 0:
            node = pop.individuals[i].genotype[pos]
            assert node in pset.functions or node in pset.terminals or node in pset.variables
            pos += 1
