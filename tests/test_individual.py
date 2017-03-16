import sys
sys.path.append('../')

import numpy as np
import pygenome.individual as individual

# tests
def test_integer_chromossome_default():
    np.random.seed(5389)
    # array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    chromossome = individual.integer_chromossome(10)
    assert np.array_equal(chromossome, np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0]))

def test_integer_chromossome_values():
    np.random.seed(5389)
    # array([-2,  3, -4, -1,  5,  5, -1,  4,  4,  0])
    chromossome = individual.integer_chromossome(10, -5, 5)
    assert np.array_equal(chromossome, np.array([-2,  3, -4, -1,  5,  5, -1,  4,  4,  0]))


