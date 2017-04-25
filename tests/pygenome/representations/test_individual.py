import numpy as np
import pygenome as pg

def test_make_individual1():
    ind = pg.Individual()
    assert type(ind) is pg.Individual
    assert ind.fitness == None
    assert ind.genotype == None
    assert ind.run_eval == True

def test_make_individual2():
    ind = pg.Individual(fitness=3, genotype=[0, 1, 0, 1, 1])
    assert type(ind) is pg.Individual
    assert ind.fitness == 3
    assert ind.genotype == [0, 1, 0, 1, 1]
    assert ind.run_eval == True
    
def test_make_individual_tree():
    ind = pg.TreeIndividual()
    assert type(ind) is pg.TreeIndividual
    assert ind.fitness == None
    assert ind.genotype == None
    assert ind.run_eval == True
    assert ind.depth == None
    assert ind.nodes == None

def test_clone_individual():
    ind1 = pg.Individual(fitness=3, genotype=[0, 1, 0, 1, 1])
    ind2 = ind1.clone()
    assert ind1.fitness == ind2.fitness
    assert ind1.genotype == ind2.genotype
    assert ind1.run_eval == ind2.run_eval
    
    ind1.fitness = 2
    ind1.genotype = [1, 0, 1, 0, 0]
    ind1.run_eval = False
    assert ind2.fitness == 3
    assert ind2.genotype == [0, 1, 0, 1, 1]
    assert ind2.run_eval is True
