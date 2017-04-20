import sys
sys.path.append('../')

import numpy as np
import pygenome as pg
import operator as op


def test_tree_crossover():
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
    