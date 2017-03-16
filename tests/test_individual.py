import sys
sys.path.append('../')

import numpy as np
import pygenome.individual as individual


def test_integer_chromossome():
    # with default values, binary chromossome
    np.random.seed(5389)
    # array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    chromossome = individual.integer_chromossome(10)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) == 0
    assert np.amax(chromossome) == 1
    assert np.array_equal(chromossome, np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0]))
    
    # with min/max values, integer chromossome
    np.random.seed(5389)
    # array([-2,  3, -4, -1,  5,  5, -1,  4,  4,  0])
    chromossome = individual.integer_chromossome(10, -5, 5)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) >= -5
    assert np.amax(chromossome) <= 5
    assert np.array_equal(chromossome, np.array([-2,  3, -4, -1,  5,  5, -1,  4,  4,  0]))
    
def test_permutation_chromossomes():
    np.random.seed(5389)
    # array([7, 9, 0, 2, 6, 4, 5, 1, 8, 3])
    chromossome = individual.permutation_chromossome(10)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) == 0
    assert np.amax(chromossome) == 9
    assert np.array_equal(np.sort(chromossome), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(chromossome, np.array([7, 9, 0, 2, 6, 4, 5, 1, 8, 3]))
    