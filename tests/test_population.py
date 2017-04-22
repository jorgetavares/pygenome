import sys
sys.path.append('../')

import numpy as np
import pygenome as pg

def test_make_population():
    pop = pg.Population()
    assert type(pop) is pg.Population
    assert pop.individuals == None
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
    