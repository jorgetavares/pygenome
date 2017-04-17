import sys
sys.path.append('../')

import numpy as np
import pygenome as pg
import operator as op


def test_tree_point_mutation():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    pop = pg.make_tree_population(2, pset, 2, 8, 150, init_method=pg.full_tree)
    i1 = pop.individuals[0].clone()
    i1m = pg.tree_point_mutation(i1, pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert i1.depth == i1m.depth
    assert i1.nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([1, 2, 1, 1, 2, 1, 6, 3, 2, 4, 5, 2, 1, 6, 3, 1, 5, 3, 2, 1, 2, 4, 6,
       2, 5, 4, 2, 2, 6, 3, 1, 3, 6, 2, 1, 1, 1, 3, 5, 1, 6, 5, 1, 2, 6, 6,
       2, 5, 6, 2, 1, 2, 3, 6, 1, 6, 3, 1, 2, 3, 5, 1, 3, 3, 2, 2, 2, 1, 1,
       4, 6, 2, 3, 4, 1, 1, 6, 4, 1, 4, 5, 1, 1, 2, 3, 6, 1, 3, 6, 2, 2, 6,
       3, 2, 4, 4, 2, 2, 1, 1, 6, 4, 2, 6, 4, 2, 2, 5, 4, 1, 6, 5, 2, 1, 2,
       6, 3, 2, 6, 3, 2, 1, 4, 5, 1, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert i1m_str == 'add(sub(add(add(sub(add(x, 1), sub(2, 3)), sub(add(x, 1), add(3, 1))), sub(add(sub(2, x), sub(3, 2)), sub(sub(x, 1), add(1, x)))), sub(add(add(add(1, 3), add(x, 3)), add(sub(x, x), sub(3, x))), sub(add(sub(1, x), add(x, 1)), add(sub(1, 3), add(1, 1))))), sub(sub(sub(add(add(2, x), sub(1, 2)), add(add(x, 2), add(2, 3))), add(add(sub(1, x), add(1, x)), sub(sub(x, 1), sub(2, 2)))), sub(sub(add(add(x, 2), sub(x, 2)), sub(sub(3, 2), add(x, 3))), sub(add(sub(x, 1), sub(x, 1)), sub(add(2, 3), add(x, 2))))))'
    