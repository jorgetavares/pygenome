import sys
sys.path.append('../')

import numpy as np
import pygenome as pg
import operator as op

def protected_div(a, b):
    if b > 0:
        return op.truediv(a, b)
    else:
        return b
        

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

def test_count_tree_internals():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    tree1 = np.array([1, 2, 1, 5, 1, 2, 2, 6, 5, 2, 3, 4, 2, 2, 4, 4, 2, 6, 6, 2, 3, 3, 1,
       2, 5, 5, 1, 5, 6, 0, 0, 0])
    depth, nodes = pg.count_tree_internals(pset, tree1)
    assert depth == 7
    assert nodes == 29
    tree2 = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0])
    depth, nodes = pg.count_tree_internals(pset, tree2)
    assert depth == 1
    assert nodes == 1

def test_count_tree_internals_typed():
    np.random.seed(42)
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, types=[int, int, int])
    pset.addFunction(op.sub, 2, types=[int, int, int])
    pset.addFunction(op.mul, 2, types=[int, int, int])
    pset.addFunction(protected_div, 2, types=[float, float, float])
    num_constants = 5
    for i in range(num_constants):
        pset.addTerminal(np.random.randint(-5, 5), types=[int])
    for i in range(num_constants):
        pset.addTerminal(np.random.uniform(), types=[float])
    pset.addVariable("x", types=[int])
    tree1 = np.array([ 3,  2,  1,  6,  8,  2,  1,  3,  7,  6,  3,  7,  8,  8,  1,  1,  8,
        6,  2, 14,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
    depth, nodes = pg.count_tree_internals(pset, tree1)
    assert depth == 6
    assert nodes == 21

    np.random.seed(42)
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, types=[int, int, int])
    pset.addFunction(op.sub, 2, types=[int, float, float])
    pset.addFunction(op.mul, 2, types=[float, int, int])
    pset.addFunction(protected_div, 2, types=[float, float, float])
    num_constants = 5
    for i in range(num_constants):
        pset.addTerminal(np.random.randint(-5, 5), types=[int])

    for i in range(num_constants):
        pset.addTerminal(np.random.uniform(), types=[float])
    pset.addVariable("x", types=[int])
    tree2 = np.array([ 1,  1,  1,  7,  6,  2,  3,  6,  8, 12,  2,  3,  7,  8,  3,  8,  5,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
    depth, nodes = pg.count_tree_internals(pset, tree2)
    assert depth == 5
    assert nodes == 17
