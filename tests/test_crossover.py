import sys
sys.path.append('../')

import numpy as np
import pygenome.crossover as cx


def test_partially_match_crossover():
    np.random.seed(42)
    g1 = np.array([4, 3, 5, 7, 8, 1, 2, 0, 9, 6])
    g2 = np.array([8, 9, 1, 3, 6, 5, 7, 4, 2, 0])
    o1, o2 = cx.partially_match_crossover(g1, g2)
    assert np.array_equal(o1, np.array([4, 7, 8, 3, 6, 5, 1, 2, 0, 9])) 
    assert np.array_equal(o2, np.array([9, 3, 6, 7, 8, 1, 5, 4, 2, 0])) 
    