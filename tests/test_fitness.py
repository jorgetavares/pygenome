import sys
sys.path.append('../')

import numpy as np
import pygenome.fitness as fitness


def test_onemax():
    assert fitness.onemax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == 10
    assert fitness.onemax(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])) == 5
    assert fitness.onemax(np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0])) == 3
    assert fitness.onemax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) == 0


def test_sphere_model():
    assert fitness.sphere_model(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])) == 0.0
    assert fitness.sphere_model(np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])) == 10.0
    assert fitness.sphere_model(np.array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])) == 40.0
    