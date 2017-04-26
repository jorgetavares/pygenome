import numpy as np
import pygenome as pg

def test_mean_squared_error1():
    x = np.array([x for x in range(100)])        
    y = np.array([i * i for i in x ])
    assert pg.mean_squared_error(x, x) == 0.0
    assert pg.mean_squared_error(x, y) == 323400.0
