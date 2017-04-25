import numpy as np
import pygenome as pg


def test_integer_chromossome():
    # with default values, binary chromossome
    np.random.seed(5389)
    # array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    chromossome = pg.integer_chromossome(10)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) == 0
    assert np.amax(chromossome) == 1
    assert np.array_equal(chromossome, np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0]))
    # with min/max values, integer chromossome
    np.random.seed(5389)
    # array([-2,  3, -4, -1,  5,  5, -1,  4,  4,  0])
    chromossome = pg.integer_chromossome(10, low=-5, high=5)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) >= -5
    assert np.amax(chromossome) <= 5
    assert np.array_equal(chromossome, np.array([-2,  3, -4, -1,  5,  5, -1,  4,  4,  0]))
    
def test_permutation_chromossome():
    np.random.seed(5389)
    # array([7, 9, 0, 2, 6, 4, 5, 1, 8, 3])
    chromossome = pg.permutation_chromossome(10)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) == 0
    assert np.amax(chromossome) == 9
    assert np.array_equal(np.sort(chromossome), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(chromossome, np.array([7, 9, 0, 2, 6, 4, 5, 1, 8, 3]))
    
def test_uniform_chromossome():
    # with default values, U(0,1)
    np.random.seed(5389)
    chromossome = pg.uniform_chromossome(10)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) >= 0.0
    assert np.amax(chromossome) < 1.0
    # with min/max values, U(min,max)
    np.random.seed(5389)
    chromossome = pg.uniform_chromossome(10, low=-5.0, high=5.0)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) >= -5.0
    assert np.amax(chromossome) < 5.0

def test_normal_chromossome():
    # with default values, N(0,1)
    np.random.seed(5389)
    chromossome = pg.normal_chromossome(10)
    assert np.size(chromossome) == 10
    # with mean/sigma values, N(mean,sigma)
    np.random.seed(5389)
    chromossome = pg.normal_chromossome(10, mean=-5.0, sigma=5.0)
    assert np.size(chromossome) == 10
    