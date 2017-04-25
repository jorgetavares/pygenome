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

def test_uniform_mutation1():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998, 0.78517596,  0.19967378,  0.51423444,  0.59241457,  0.04645041]))
    m1 = pg.uniform_mutation(i1, gene_rate=0.0, low=0.0, high=1.0)
    assert m1.run_eval is True
    assert np.isclose(m1.genotype.all(), np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998, 0.78517596,  0.19967378,  0.51423444,  0.59241457,  0.04645041]).all()) 

def test_uniform_mutation2():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998, 0.78517596,  0.19967378,  0.51423444,  0.59241457,  0.04645041]))
    m1 = pg.uniform_mutation(i1, gene_rate=1.0, low=0.0, high=1.0)
    assert m1.run_eval is True
    assert np.isclose(m1.genotype.all(), np.array([ 0.95071431,  0.59865848,  0.15599452,  0.86617615,  0.70807258, 0.96990985,  0.21233911,  0.18340451,  0.52475643,  0.29122914]).all()) 

def test_uniform_mutation3():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998, 0.78517596,  0.19967378,  0.51423444,  0.59241457,  0.04645041]))
    m1 = pg.uniform_mutation(i1, gene_rate=None, low=0.0, high=1.0)
    assert m1.run_eval is True
    assert np.isclose(m1.genotype.all(), np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998, 0.78517596,  0.86617615,  0.51423444,  0.59241457,  0.96990985]).all()) 

def test_sigma_check():
    assert np.isclose(pg.sigma_check(0.0, epsilon=1e-08), 1e-08) is True
    assert np.isclose(pg.sigma_check(-0.5, epsilon=1e-08), 1e-08) is True
    assert np.isclose(pg.sigma_check(0.5, epsilon=1e-08), 0.5) is True

def test_apply_global_mutation():
    np.random.seed(42)
    size = 30
    ind_size = 100
    operator = pg.uncorrelated_one_step_mutation
    pool_size = 100
    pop = pg.make_uniform_population(size, ind_size)
    original_pop = pop.clone()
    pop = pg.apply_global_mutation(pop, pool_size, operator)
    assert pop.size == pool_size
    for i in range(pop.size):
        assert pop.individuals[i].run_eval is True

def test_uncorrelated_one_step_mutation():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998, 0.78517596,  0.19967378,  0.51423444,  0.59241457,  0.04645041]))
    m1 = pg.uncorrelated_one_step_mutation(i1)
    assert np.isclose(m1.genotype.all(), np.array([ 0.60427399,  0.17499665,  0.37562893,  0.35352682,  0.44323586, 0.87173989,  0.24174041,  0.48850039,  0.62215478,  0.05481461]).all())

def test_uncorrelated_n_steps_mutation():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([ 0.61185289,  0.13949386,  0.29214465,  0.36636184,  0.45606998, 0.78517596,  0.19967378,  0.51423444,  0.59241457,  0.04645041]))
    m1 = pg.uncorrelated_n_steps_mutation(i1)
    assert np.isclose(m1.genotype.all(), np.array([ 1.97088495,  0.38304596, -0.28830838,  0.70303098,  0.43352264, 0.86057563,  0.31735872,  1.23638915,  0.6205196 ,  0.04865446]).all())
