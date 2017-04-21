import sys
sys.path.append('../')

import numpy as np
import pygenome as pg


def test_partially_match_crossover():
    np.random.seed(42)
    i1 = pg.Individual(genotype=np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6]))
    i2 = pg.Individual(genotype=np.array([8, 9, 1, 3, 6, 5, 7, 4, 2, 0]))
    o1, o2 = pg.partially_match_crossover(i1, i2)
    assert np.array_equal(o1.genotype, np.array([4, 7, 8, 3, 6, 5, 1, 2, 0, 9])) 
    assert np.array_equal(o2.genotype, np.array([9, 3, 6, 7, 8, 1, 5, 4, 2, 0])) 
    