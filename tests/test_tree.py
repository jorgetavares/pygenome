import sys
sys.path.append('../')

import numpy as np
import pygenome as pg
import operator as op


def test_transverse_tree():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    tree = np.array([1, 3, 1, 3, 4, 0, 0])
    assert pg.transverse_tree(pset, tree, 0) == 5
    assert pg.transverse_tree(pset, tree, 1) == 2
    assert pg.transverse_tree(pset, tree, 2) == 5
