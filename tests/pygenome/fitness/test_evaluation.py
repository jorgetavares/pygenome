import numpy as np
import pygenome as pg

def test_Fitness():
    fit = pg.Fitness()
    assert fit.value == None
    fit = pg.Fitness(value=1.0)
    assert fit.value == 1.0
