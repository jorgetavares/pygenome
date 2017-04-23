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


def test_tree_crossover1():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    pop = pg.make_tree_population(2, pset, 4, 6, init_method=pg.full_tree)
    i1 = pop.individuals[0].clone()
    i2 = pop.individuals[1].clone()
    i1m, i2m = pg.tree_crossover(i1, i2, pset=pset)
    i1m_str = pg.interpreter(pset, i1m.genotype)
    i2m_str = pg.interpreter(pset, i2m.genotype)

    assert i1m.depth == 7
    assert i1m.nodes == 29
    assert np.array_equal(i1m.genotype, np.array([1, 2, 1, 5, 1, 2, 2, 6, 5, 2, 3, 4, 2, 2, 4, 4, 2, 6, 6, 2, 3, 3, 1,
       2, 5, 5, 1, 5, 6, 0, 0, 0]))
    assert i1m_str == 'add(sub(add(3, add(sub(sub(x, 3), sub(1, 2)), sub(sub(2, 2), sub(x, x)))), sub(1, 1)), add(sub(3, 3), add(3, x)))'
    
    assert i2m.depth == 1
    assert i2m.nodes == 1
    assert np.array_equal(i2m.genotype, np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert i2m_str == '3'

def test_tree_crossover2():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addFunction(op.mul, 2)
    pset.addFunction(protected_div, 2)
    num_constants = 10
    for i in range(num_constants):
        pset.addTerminal(np.random.randint(-5, 5))
    pset.addVariable("x")
    
    t1 = np.array([3, 9, 1, 1, 5, 7, 6, 0, 0, 0])
    t2 = np.array([6, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    i1 = pg.TreeIndividual(tree=t1, nodes=7)
    i2 = pg.TreeIndividual(tree=t2, nodes=1)
    o1, o2 = pg.tree_crossover(i1, i2, pset=pset)
    o1_str = pg.interpreter(pset, o1.genotype)
    o2_str = pg.interpreter(pset, o2.genotype)
    
    assert o1.depth == 3
    assert o1.nodes == 5
    assert np.array_equal(o1.genotype, np.array([3, 9, 1, 6, 6, 0, 0, 0, 0, 0]))
    assert o1_str == 'mul(4, add(-2, -2))'
    
    assert o2.depth == 2
    assert o2.nodes == 3
    assert np.array_equal(o2.genotype, np.array([1, 5, 7, 0, 0, 0, 0, 0, 0, 0]))
    assert o2_str == 'add(1, 2)'


def test_tree_crossover_typed1():
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

    pop = pg.make_tree_population(2, pset, 4, 6, init_method=pg.full_tree)
    o1, o2 = pg.tree_crossover(pop.individuals[0], pop.individuals[1], pset=pset)
    o1_str = pg.interpreter(pset, o1.genotype)
    o2_str = pg.interpreter(pset, o2.genotype)

    assert o1.depth == 6
    assert o1.nodes == 21
    assert np.array_equal(o1.genotype, np.array([ 3,  2,  1,  6,  8,  2,  1,  3,  7,  6,  3,  7,  8,  8,  1,  1,  8,
        6,  2, 14,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert o1_str == 'mul(sub(add(-2, -1), sub(add(mul(2, -2), mul(2, -1)), -1)), add(add(-1, -2), sub(x, -1)))'
    
    assert o2.depth == 4
    assert o2.nodes == 9
    assert np.array_equal(o2.genotype, np.array([ 1,  6,  3,  1,  7, 14,  3, 14,  5,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert o2_str == 'add(-2, mul(add(2, x), mul(x, 1)))'

def test_tree_crossover_typed2():
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

    pop = pg.make_tree_population(2, pset, 4, 6, init_method=pg.full_tree)
    o1, o2 = pg.tree_crossover(pop.individuals[0], pop.individuals[1], pset=pset)
    o1_str = pg.interpreter(pset, o1.genotype)
    o2_str = pg.interpreter(pset, o2.genotype)

    assert o1.depth == 4
    assert o1.nodes == 13
    assert np.array_equal(o1.genotype, np.array([ 3,  2, 12,  4, 10, 12,  1,  1,  8,  6,  2, 13, 12,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert o1_str == 'mul(sub(0.33370861113902184, protected_div(0.09997491581800289, 0.33370861113902184)), add(add(-1, -2), sub(0.14286681792194078, 0.33370861113902184)))'
    
    assert o2.depth == 5
    assert o2.nodes == 17
    assert np.array_equal(o2.genotype, np.array([ 1,  1,  1,  7,  6,  2,  3,  6,  8, 12,  2,  3,  7,  8,  3,  8,  5,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert o2_str == 'add(add(add(2, -2), sub(mul(-2, -1), 0.33370861113902184)), sub(mul(2, -1), mul(-1, 1)))'
