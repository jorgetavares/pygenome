import numpy as np
import pygenome as pg


def test_onemax():
    assert pg.onemax(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == 10
    assert pg.onemax(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])) == 5
    assert pg.onemax(np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0])) == 3
    assert pg.onemax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) == 0


def test_sphere_model():
    assert pg.sphere_model(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])) == 0.0
    assert pg.sphere_model(np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])) == 10.0
    assert pg.sphere_model(np.array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])) == 40.0
    