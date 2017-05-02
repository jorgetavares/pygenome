import numpy as np
import pygenome as pg
import operator as op

def test_make_individual1():
    ind = pg.Individual()
    assert type(ind) is pg.Individual
    assert ind.fitness is None
    assert ind.genotype is None
    assert ind.run_eval is True

def test_make_individual2():
    ind = pg.Individual(fitness=pg.Fitness(value=3), genotype=[0, 1, 0, 1, 1])
    assert type(ind) is pg.Individual
    assert ind.fitness.value == 3
    assert ind.genotype == [0, 1, 0, 1, 1]
    assert ind.run_eval is True
    
def test_make_individual_tree():
    ind = pg.TreeIndividual()
    assert type(ind) is pg.TreeIndividual
    assert ind.fitness is None
    assert ind.genotype is None
    assert ind.run_eval is True
    assert ind.depth is None
    assert ind.nodes is None

def test_clone_individual():
    ind1 = pg.Individual(fitness=pg.Fitness(value=3), genotype=[0, 1, 0, 1, 1])
    ind2 = ind1.clone()
    assert ind1.fitness.equal(ind2.fitness) is True
    assert ind1.genotype == ind2.genotype
    assert ind1.run_eval == ind2.run_eval
    
    ind1.fitness = 2
    ind1.genotype = [1, 0, 1, 0, 0]
    ind1.run_eval = False
    assert ind2.fitness.value == 3
    assert ind2.genotype == [0, 1, 0, 1, 1]
    assert ind2.run_eval is True

def test_equal_individual():
    ind1 = pg.Individual(fitness=pg.Fitness(value=3), genotype=[0, 1, 0, 1, 1])
    ind2 = ind1.clone()
    assert ind1.equal(ind2) is True
    ind3 = pg.Individual(fitness=pg.Fitness(value=3), genotype=[0, 1, 1, 1, 1])
    assert ind1.equal(ind3) is False
    ind4 = pg.Individual(fitness=pg.Fitness(value=2), genotype=[0, 1, 0, 1, 1])
    assert ind1.equal(ind4) is False
    ind5 = pg.Individual(fitness=pg.Fitness(value=3), genotype=[0, 1, 0, 1, 1])
    ind5.run_eval = False
    assert ind1.equal(ind5) is False

def test_equal_tree_individual():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    pop = pg.make_tree_population(1, pset, 7, 8, init_method=pg.full_tree)
    i1 = pop.individuals[0]
    i2 = i1.clone()
    assert i1.equal(i2) is True
    pop = pg.make_tree_population(1, pset, 7, 8, init_method=pg.full_tree)
    i3 = pop.individuals[0]
    assert i1.equal(i3) is False
