import numpy as np
import pygenome as pg


def test_binary_chromossome():
    np.random.seed(42)
    # array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    chromossome = pg.integer_chromossome(10)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) == 0
    assert np.amax(chromossome) == 1
    assert np.array_equal(chromossome, np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1]))

def test_integer_chromossome():    
    np.random.seed(42)
    # array([ 1, -2,  5,  2, -1,  1,  4, -3,  1,  5])
    chromossome = pg.integer_chromossome(10, low=-5, high=5)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) >= -5
    assert np.amax(chromossome) <= 5
    assert np.array_equal(chromossome, np.array([ 1, -2,  5,  2, -1,  1,  4, -3,  1,  5]))
    
def test_permutation_chromossome():
    np.random.seed(42)
    # array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    chromossome = pg.permutation_chromossome(10)
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) == 0
    assert np.amax(chromossome) == 9
    assert np.array_equal(np.sort(chromossome), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(chromossome, np.array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6]))
    
def test_uniform_chromossome():
    np.random.seed(42)
    chromossome = pg.uniform_chromossome(10)
    # array([ 0.37454012,  0.95071431,  0.73199394,  0.59865848,  0.15601864,
    #    0.15599452,  0.05808361,  0.86617615,  0.60111501,  0.70807258])
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) >= 0.0
    assert np.amax(chromossome) < 1.0
    assert np.allclose(chromossome, np.array([ 0.37454012,  0.95071431,  0.73199394,  0.59865848,  0.15601864, 0.15599452,  0.05808361,  0.86617615,  0.60111501,  0.70807258]))

def test_uniform_chromossome_nondefault():
    np.random.seed(42)
    chromossome = pg.uniform_chromossome(10, low=-5.0, high=5.0)
    # array([-1.25459881,  4.50714306,  2.31993942,  0.98658484, -3.4398136 , -3.4400548 , -4.41916388,  3.66176146,  1.01115012,  2.08072578])
    assert np.size(chromossome) == 10
    assert np.amin(chromossome) >= -5.0
    assert np.amax(chromossome) < 5.0
    assert np.allclose(chromossome, np.array([-1.25459881,  4.50714306,  2.31993942,  0.98658484, -3.4398136 ,
       -3.4400548 , -4.41916388,  3.66176146,  1.01115012,  2.08072578]))

def test_normal_chromossome():
    np.random.seed(42)
    chromossome = pg.normal_chromossome(10)
    # array([ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337, -0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004])
    assert np.size(chromossome) == 10
    assert np.allclose(chromossome, np.array([ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,
       -0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004]))

def test_normal_chromossome_nondefault():
    np.random.seed(42)
    chromossome = pg.normal_chromossome(10, mean=-5.0, sigma=5.0)
    # array([-2.51642923, -5.69132151, -1.76155731,  2.61514928, -6.17076687, -6.17068478,  2.89606408, -1.16282635, -7.34737193, -2.28719978])
    assert np.size(chromossome) == 10
    assert np.allclose(chromossome, np.array([-2.51642923, -5.69132151, -1.76155731,  2.61514928, -6.17076687,
       -6.17068478,  2.89606408, -1.16282635, -7.34737193, -2.28719978]))
