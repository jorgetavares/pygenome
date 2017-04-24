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

def test_interpreter_str():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    tree = np.array([1, 2, 1, 1, 1, 2, 3, 3, 1, 4, 5, 1, 1, 5, 6, 1, 6, 6, 2, 1, 2, 3, 4,
       2, 6, 4, 2, 2, 6, 6, 1, 3, 6, 2, 2, 1, 2, 3, 3, 1, 5, 5, 2, 2, 6, 6,
       2, 5, 4, 2, 1, 2, 5, 6, 1, 6, 6, 1, 1, 3, 5, 1, 3, 3, 1, 2, 2, 1, 2,
       4, 4, 2, 3, 4, 1, 2, 6, 6, 1, 6, 5, 2, 1, 2, 5, 5, 2, 3, 4, 2, 2, 6,
       4, 2, 4, 4, 2, 2, 2, 1, 5, 4, 2, 6, 4, 2, 2, 6, 4, 1, 6, 5, 2, 2, 1,
       6, 3, 2, 6, 3, 2, 1, 4, 5, 1, 6, 4, 0])
    tree_str = pg.interpreter(pset, tree)
    assert tree_str == 'add(sub(add(add(add(sub(1, 1), add(2, 3)), add(add(3, x), add(x, x))), sub(add(sub(1, 2), sub(x, 2)), sub(sub(x, x), add(1, x)))), sub(sub(add(sub(1, 1), add(3, 3)), sub(sub(x, x), sub(3, 2))), sub(add(sub(3, x), add(x, x)), add(add(1, 3), add(1, 1))))), add(sub(sub(add(sub(2, 2), sub(1, 2)), add(sub(x, x), add(x, 3))), sub(add(sub(3, 3), sub(1, 2)), sub(sub(x, 2), sub(2, 2)))), sub(sub(sub(add(3, 2), sub(x, 2)), sub(sub(x, 2), add(x, 3))), sub(sub(add(x, 1), sub(x, 1)), sub(add(2, 3), add(x, 2))))))'

def test_interpreter_str_typed():
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [float, int, int])
    pset.addFunction(op.sub, 2, [int, float, float])
    pset.addFunction(op.mul, 2, [float, int, int])
    pset.addTerminal(1, [int])
    pset.addTerminal(2, [int])
    pset.addTerminal(3, [int])
    pset.addTerminal(4.0, [float])
    pset.addTerminal(5.0, [float])
    pset.addTerminal(6.0, [float])
    pset.addVariable("x", [float])
    pset.addVariable("y", [int])
    tree = np.array([ 1,  2,  3,  2,  1,  2,  9,  9,  2, 10,  7,  1,  2,  9,  8,  2,  9,
        9,  2,  1,  2,  9, 10,  2,  7, 10,  3,  2, 10,  9,  2,  8,  7,  1,
        2,  3,  2, 10,  9,  2,  9,  8,  1,  2, 10,  7,  2,  7, 10,  2,  3,
        2,  7,  9,  2, 10,  7,  1,  2,  9,  9,  2,  8,  8,  2,  3,  2,  3,
        2, 10, 10,  2,  9,  9,  3,  2,  8,  8,  2,  9, 10,  2,  3,  2, 10,
       10,  2,  7,  8,  1,  2,  9,  8,  2,  7,  7,  3,  2,  3,  2, 10,  7,
        2, 10,  8,  3,  2,  7,  9,  2,  9,  8,  2,  3,  2, 10, 10,  2,  9,
       10,  1,  2, 10,  7,  2, 10,  9,  0])
    tree_str = pg.interpreter(pset, tree)
    assert tree_str == 'add(sub(mul(sub(add(sub(6.0, 6.0), sub(x, 4.0)), add(sub(6.0, 5.0), sub(6.0, 6.0))), sub(add(sub(6.0, x), sub(4.0, x)), mul(sub(x, 6.0), sub(5.0, 4.0)))), add(sub(mul(sub(x, 6.0), sub(6.0, 5.0)), add(sub(x, 4.0), sub(4.0, x))), sub(mul(sub(4.0, 6.0), sub(x, 4.0)), add(sub(6.0, 6.0), sub(5.0, 5.0))))), sub(mul(sub(mul(sub(x, x), sub(6.0, 6.0)), mul(sub(5.0, 5.0), sub(6.0, x))), sub(mul(sub(x, x), sub(4.0, 5.0)), add(sub(6.0, 5.0), sub(4.0, 4.0)))), mul(sub(mul(sub(x, 4.0), sub(x, 5.0)), mul(sub(4.0, 6.0), sub(6.0, 5.0))), sub(mul(sub(x, x), sub(6.0, x)), add(sub(x, 4.0), sub(x, 6.0))))))'

def test_interpreter_run():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addTerminal(4)
    tree = np.array([1, 2, 1, 1, 1, 2, 3, 3, 1, 4, 5, 1, 1, 5, 6, 1, 6, 6, 2, 1, 2, 3, 4,
       2, 6, 4, 2, 2, 6, 6, 1, 3, 6, 2, 2, 1, 2, 3, 3, 1, 5, 5, 2, 2, 6, 6,
       2, 5, 4, 2, 1, 2, 5, 6, 1, 6, 6, 1, 1, 3, 5, 1, 3, 3, 1, 2, 2, 1, 2,
       4, 4, 2, 3, 4, 1, 2, 6, 6, 1, 6, 5, 2, 1, 2, 5, 5, 2, 3, 4, 2, 2, 6,
       4, 2, 4, 4, 2, 2, 2, 1, 5, 4, 2, 6, 4, 2, 2, 6, 4, 1, 6, 5, 2, 2, 1,
       6, 3, 2, 6, 3, 2, 1, 4, 5, 1, 6, 4, 0])
    tree_run = pg.interpreter(pset, tree, run=True)
    assert tree_run == 20

def test_interpreter_run_typed():
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [float, int, int])
    pset.addFunction(op.sub, 2, [int, float, float])
    pset.addFunction(op.mul, 2, [float, int, int])
    pset.addTerminal(1, [int])
    pset.addTerminal(2, [int])
    pset.addTerminal(3, [int])
    pset.addTerminal(4.0, [float])
    pset.addTerminal(5.0, [float])
    pset.addTerminal(6.0, [float])
    pset.addTerminal(7.0, [float])
    pset.addTerminal(8, [int])
    tree = np.array([ 1,  2,  3,  2,  1,  2,  9,  9,  2, 10,  7,  1,  2,  9,  8,  2,  9,
        9,  2,  1,  2,  9, 10,  2,  7, 10,  3,  2, 10,  9,  2,  8,  7,  1,
        2,  3,  2, 10,  9,  2,  9,  8,  1,  2, 10,  7,  2,  7, 10,  2,  3,
        2,  7,  9,  2, 10,  7,  1,  2,  9,  9,  2,  8,  8,  2,  3,  2,  3,
        2, 10, 10,  2,  9,  9,  3,  2,  8,  8,  2,  9, 10,  2,  3,  2, 10,
       10,  2,  7,  8,  1,  2,  9,  8,  2,  7,  7,  3,  2,  3,  2, 10,  7,
        2, 10,  8,  3,  2,  7,  9,  2,  9,  8,  2,  3,  2, 10, 10,  2,  9,
       10,  1,  2, 10,  7,  2, 10,  9,  0])
    tree_run = pg.interpreter(pset, tree, run=True)
    assert tree_run == 27.0

def test_interpreter_run_inputs():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    tree = np.array([1, 2, 1, 1, 1, 2, 3, 3, 1, 4, 5, 1, 1, 5, 6, 1, 6, 6, 2, 1, 2, 3, 4,
       2, 6, 4, 2, 2, 6, 6, 1, 3, 6, 2, 2, 1, 2, 3, 3, 1, 5, 5, 2, 2, 6, 6,
       2, 5, 4, 2, 1, 2, 5, 6, 1, 6, 6, 1, 1, 3, 5, 1, 3, 3, 1, 2, 2, 1, 2,
       4, 4, 2, 3, 4, 1, 2, 6, 6, 1, 6, 5, 2, 1, 2, 5, 5, 2, 3, 4, 2, 2, 6,
       4, 2, 4, 4, 2, 2, 2, 1, 5, 4, 2, 6, 4, 2, 2, 6, 4, 1, 6, 5, 2, 2, 1,
       6, 3, 2, 6, 3, 2, 1, 4, 5, 1, 6, 4, 0])
    inputs = {"x": 4}
    tree_run = pg.interpreter(pset, tree, run=True, vars_inputs=inputs)
    assert tree_run == 20

def test_interpreter_run_typed_inputs():
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [float, int, int])
    pset.addFunction(op.sub, 2, [int, float, float])
    pset.addFunction(op.mul, 2, [float, int, int])
    pset.addTerminal(1, [int])
    pset.addTerminal(2, [int])
    pset.addTerminal(3, [int])
    pset.addTerminal(4.0, [float])
    pset.addTerminal(5.0, [float])
    pset.addTerminal(6.0, [float])
    pset.addVariable("x", [float])
    pset.addVariable("y", [int])
    inputs = {"x": 7.0, "y": 8}
    tree = np.array([ 1,  2,  3,  2,  1,  2,  9,  9,  2, 10,  7,  1,  2,  9,  8,  2,  9,
        9,  2,  1,  2,  9, 10,  2,  7, 10,  3,  2, 10,  9,  2,  8,  7,  1,
        2,  3,  2, 10,  9,  2,  9,  8,  1,  2, 10,  7,  2,  7, 10,  2,  3,
        2,  7,  9,  2, 10,  7,  1,  2,  9,  9,  2,  8,  8,  2,  3,  2,  3,
        2, 10, 10,  2,  9,  9,  3,  2,  8,  8,  2,  9, 10,  2,  3,  2, 10,
       10,  2,  7,  8,  1,  2,  9,  8,  2,  7,  7,  3,  2,  3,  2, 10,  7,
        2, 10,  8,  3,  2,  7,  9,  2,  9,  8,  2,  3,  2, 10, 10,  2,  9,
       10,  1,  2, 10,  7,  2, 10,  9,  0])
    tree_run = pg.interpreter(pset, tree, run=True, vars_inputs=inputs)
    assert tree_run == 27.0

def test_max_size_from_tree_max_depth_arity2():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addTerminal(4)
    assert pg.max_size_from_tree_max_depth(pset, 4) == 8
    assert pg.max_size_from_tree_max_depth(pset, 6) == 32
    assert pg.max_size_from_tree_max_depth(pset, 8) == 128
    assert pg.max_size_from_tree_max_depth(pset, 10) == 512
    assert pg.max_size_from_tree_max_depth(pset, 12) == 2048
    assert pg.max_size_from_tree_max_depth(pset, 14) == 8192
    assert pg.max_size_from_tree_max_depth(pset, 16) == 32768
    assert pg.max_size_from_tree_max_depth(pset, 18) == 131072
    assert pg.max_size_from_tree_max_depth(pset, 20) == 524288

def test_max_size_from_tree_max_depth_arity3():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 3)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addTerminal(4)
    assert pg.max_size_from_tree_max_depth(pset, 4) == 13
    assert pg.max_size_from_tree_max_depth(pset, 6) == 121
    assert pg.max_size_from_tree_max_depth(pset, 8) == 1093
    assert pg.max_size_from_tree_max_depth(pset, 10) == 9841
    assert pg.max_size_from_tree_max_depth(pset, 12) == 88573
    assert pg.max_size_from_tree_max_depth(pset, 14) == 797161
    assert pg.max_size_from_tree_max_depth(pset, 16) == 7174453
    assert pg.max_size_from_tree_max_depth(pset, 18) == 64570081
    assert pg.max_size_from_tree_max_depth(pset, 20) == 581130733
