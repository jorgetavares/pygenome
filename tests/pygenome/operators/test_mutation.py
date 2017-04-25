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
        assert pop.individuals[i].run_eval is True
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
        assert pop.individuals[i].run_eval is True
        assert np.array_equal(pop.individuals[i].genotype, original_pop.individuals[i].genotype)

def test_binary_flip_mutation1():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]))
    m1 = pg.binary_flip_mutation(i1, gene_rate=0.0)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1])) 

def test_binary_flip_mutation2():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]))
    m1 = pg.binary_flip_mutation(i1, gene_rate=1.0,)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0])) 

def test_binary_flip_mutation3():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]))
    m1 = pg.binary_flip_mutation(i1, gene_rate=None,)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1])) 

def test_flip_mutation1():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6]))
    m1 = pg.flip_mutation(i1, gene_rate=0.0, low=0, high=9)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6])) 

def test_flip_mutation2():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6]))
    m1 = pg.flip_mutation(i1, gene_rate=1.0, low=0, high=9)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([7, 6, 6, 7, 7, 5, 7, 5, 4, 9])) 

def test_flip_mutation3():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6]))
    m1 = pg.flip_mutation(i1, gene_rate=None, low=0, high=9)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([4, 3, 5, 7, 8, 1, 7, 0, 9, 6])) 

def test_swap_mutation1():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6]))
    m1 = pg.swap_mutation(i1, gene_rate=0.0)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6])) 

def test_swap_mutation2():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6]))
    m1 = pg.swap_mutation(i1, gene_rate=1.0)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([0, 2, 3, 4, 9, 5, 8, 1, 7, 6])) 

def test_swap_mutation3():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6]))
    m1 = pg.swap_mutation(i1, gene_rate=None)
    assert m1.run_eval is True
    assert np.array_equal(m1.genotype, np.array([4, 3, 5, 7, 8, 1, 0, 2, 9, 6]))
