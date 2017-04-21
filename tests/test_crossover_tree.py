import sys
sys.path.append('../')

import numpy as np
import pygenome as pg
import operator as op


def test_tree_crossover1():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    pop = pg.make_tree_population(2, pset, 1, 4, 50, init_method=pg.full_tree)
    i1 = pop.individuals[0].clone()
    i2 = pop.individuals[1].clone()
    i1m, i2m = pg.tree_crossover(i1, i2, pset=pset)
    i1m_str = pg.interpreter(pset, i1m.genotype)
    i2m_str = pg.interpreter(pset, i2m.genotype)

    assert i1m.depth == 7
    assert i1m.nodes == 29
    assert np.array_equal(i1m.genotype, np.array([1, 2, 1, 5, 1, 2, 2, 6, 5, 2, 3, 4, 2, 2, 4, 4, 2, 6, 6, 2, 3, 3, 1,
       2, 5, 5, 1, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0]))
    assert i1m_str == 'add(sub(add(3, add(sub(sub(x, 3), sub(1, 2)), sub(sub(2, 2), sub(x, x)))), sub(1, 1)), add(sub(3, 3), add(3, x)))'
    
    assert i2m.depth == 1
    assert i2m.nodes == 1
    assert np.array_equal(i2m.genotype, np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0]))
    assert i2m_str == '3'

def test_tree_crossover2():

    def protected_div(a, b):
        if b > 0:
            return op.truediv(a, b)
        else:
            return b

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
