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
    assert pg.max_size_from_tree_max_depth(pset, 4) == 16
    assert pg.max_size_from_tree_max_depth(pset, 6) == 64
    assert pg.max_size_from_tree_max_depth(pset, 8) == 256
    assert pg.max_size_from_tree_max_depth(pset, 10) == 1024
    assert pg.max_size_from_tree_max_depth(pset, 12) == 4096
    assert pg.max_size_from_tree_max_depth(pset, 14) == 16384
    assert pg.max_size_from_tree_max_depth(pset, 16) == 65536
    assert pg.max_size_from_tree_max_depth(pset, 18) == 262144
    assert pg.max_size_from_tree_max_depth(pset, 20) == 1048576

def test_max_size_from_tree_max_depth_arity3():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 3)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addTerminal(4)
    assert pg.max_size_from_tree_max_depth(pset, 4) == 40
    assert pg.max_size_from_tree_max_depth(pset, 6) == 364
    assert pg.max_size_from_tree_max_depth(pset, 8) == 3280
    assert pg.max_size_from_tree_max_depth(pset, 10) == 29524
    assert pg.max_size_from_tree_max_depth(pset, 12) == 265720
    assert pg.max_size_from_tree_max_depth(pset, 14) == 2391484
    assert pg.max_size_from_tree_max_depth(pset, 16) == 21523360
    assert pg.max_size_from_tree_max_depth(pset, 18) == 193710244
    assert pg.max_size_from_tree_max_depth(pset, 20) == 1743392200

def test_full_tree1():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    init_max_depth = 3
    max_depth = 6
    tree = pg.full_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([1, 2, 3, 5, 1, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

def test_full_tree2():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    init_max_depth = 8
    max_depth = 8
    tree = pg.full_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([1, 2, 1, 1, 1, 2, 1, 3, 5, 2, 5, 5, 1, 1, 6, 3, 2, 6, 6, 1, 2, 1, 4,
       6, 2, 4, 4, 2, 2, 6, 3, 1, 6, 4, 2, 1, 2, 1, 3, 5, 1, 5, 4, 2, 2, 6,
       6, 1, 4, 4, 1, 2, 1, 6, 5, 2, 6, 3, 1, 1, 5, 5, 1, 3, 5, 2, 2, 1, 2,
       2, 4, 4, 1, 4, 3, 2, 2, 6, 5, 2, 5, 6, 1, 2, 1, 5, 4, 1, 6, 4, 2, 2,
       4, 4, 2, 4, 4, 2, 2, 1, 1, 4, 4, 2, 4, 4, 2, 2, 4, 5, 2, 5, 6, 2, 1,
       2, 3, 4, 2, 3, 6, 1, 2, 5, 3, 2, 4, 3, 2, 2, 2, 1, 1, 1, 5, 3, 1, 3,
       5, 1, 2, 3, 6, 2, 6, 5, 1, 1, 1, 6, 5, 1, 3, 5, 1, 2, 5, 4, 1, 6, 5,
       1, 2, 2, 2, 3, 6, 1, 5, 4, 2, 1, 5, 6, 2, 4, 5, 1, 1, 1, 3, 5, 2, 5,
       3, 1, 2, 5, 5, 2, 5, 5, 1, 1, 1, 2, 2, 6, 3, 1, 5, 6, 1, 1, 6, 3, 2,
       6, 4, 1, 1, 1, 3, 5, 1, 3, 6, 1, 2, 5, 5, 1, 4, 6, 2, 2, 1, 2, 3, 3,
       2, 6, 6, 2, 2, 6, 4, 2, 5, 6, 2, 1, 2, 3, 5, 2, 3, 3, 1, 1, 4, 3, 2,
       3, 3, 0]))

def test_grow_tree1():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    init_max_depth = 3
    max_depth = 6
    tree = pg.grow_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

def test_grow_tree2():
    np.random.seed(45345)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    init_max_depth = 3
    max_depth = 6
    tree = pg.grow_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([1, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

def test_grow_tree3():
    np.random.seed(45345)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    init_max_depth = 2
    max_depth = 2
    tree = pg.full_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([1, 6, 5, 0]))

#####
def test_full_tree1_typed():
    np.random.seed(42)
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [int, int, int])
    pset.addFunction(op.sub, 2, [int, int, int])
    pset.addTerminal(1, [int])
    pset.addTerminal(2, [int])
    pset.addTerminal(3, [int])
    pset.addVariable("x", [int])
    init_max_depth = 3
    max_depth = 6
    tree = pg.full_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([1, 2, 3, 5, 1, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

def test_full_tree2_typed():
    np.random.seed(42)
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [int, int, int])
    pset.addFunction(op.sub, 2, [float, int, int])
    pset.addFunction(op.mul, 2, [int, float, float])
    pset.addFunction(op.truediv, 2, [float, float, float])
    pset.addTerminal(1, [int])
    pset.addTerminal(2, [int])
    pset.addTerminal(3, [int])
    pset.addTerminal(4.0, [float])
    pset.addTerminal(5.0, [float])
    pset.addTerminal(6.0, [float])
    pset.addVariable("x", [float])
    pset.addVariable("y", [int])
    init_max_depth = 7
    max_depth = 7
    tree = pg.full_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([ 3,  4,  2,  1,  1,  3,  8,  8,  1,  6,  7,  1,  1,  7, 12,  1, 12,
       12,  3,  2,  3,  8,  9,  3, 11,  9,  4,  4, 11, 11,  2,  5, 12,  4,
        4,  2,  3,  8,  8,  1,  7,  7,  4,  4, 11, 11,  4, 10,  9,  4,  2,
        3, 10, 11,  1, 12, 12,  2,  1,  5,  7,  1,  5,  5,  2,  3,  4,  2,
        3,  9,  9,  3,  8,  9,  2,  3, 11, 11,  1, 12,  7,  4,  2,  3, 10,
       10,  3,  8, 11,  4,  4, 11,  9,  4,  9,  9,  3,  4,  4,  2,  7,  6,
        4, 11,  9,  4,  4, 11,  9,  2, 12,  7,  4,  4,  2, 12,  5,  4, 11,
        8,  4,  2,  6,  7,  2, 12,  6,  0]))

def test_grow_tree1():
    np.random.seed(45345)
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [int, int, int])
    pset.addFunction(op.sub, 2, [int, int, int])
    pset.addTerminal(1, [int])
    pset.addTerminal(2, [int])
    pset.addTerminal(3, [int])
    pset.addVariable("x", [int])
    init_max_depth = 3
    max_depth = 6
    tree = pg.grow_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([1, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

def test_grow_tree2():
    np.random.seed(12345)
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [int, int, int])
    pset.addFunction(op.sub, 2, [float, int, int])
    pset.addFunction(op.mul, 2, [int, float, float])
    pset.addFunction(op.truediv, 2, [float, float, float])
    pset.addTerminal(1, [int])
    pset.addTerminal(2, [int])
    pset.addTerminal(3, [int])
    pset.addTerminal(4.0, [float])
    pset.addTerminal(5.0, [float])
    pset.addTerminal(6.0, [float])
    pset.addVariable("x", [float])
    pset.addVariable("y", [int])
    init_max_depth = 4
    max_depth = 4
    tree = pg.grow_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([ 3, 11, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))

def test_grow_tree3():
    np.random.seed(12345)
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [int, int, int])
    pset.addFunction(op.sub, 2, [float, int, int])
    pset.addFunction(op.mul, 2, [int, float, float])
    pset.addFunction(op.truediv, 2, [float, float, float])
    pset.addTerminal(1, [int])
    pset.addTerminal(2, [int])
    pset.addTerminal(3, [int])
    pset.addTerminal(4.0, [float])
    pset.addTerminal(5.0, [float])
    pset.addTerminal(6.0, [float])
    pset.addVariable("x", [float])
    pset.addVariable("y", [int])
    init_max_depth = 2
    max_depth = 2
    tree = pg.grow_tree(pset, init_max_depth, max_depth)
    assert np.array_equal(tree, np.array([3, 9, 9, 0]))
