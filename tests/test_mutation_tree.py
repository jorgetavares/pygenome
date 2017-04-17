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
    
def test_tree_point_mutation_st1():
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
    pop = pg.make_tree_population(2, pset, 3, 8, 150, init_method=pg.full_tree)
    i1 = pop.individuals[0].clone()
    i1m = pg.tree_point_mutation(i1, pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)
    
    assert i1.depth == i1m.depth
    assert i1.nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([ 3,  4,  2,  1,  1, 12,  5,  1,  7,  6,  1,  1,  7,  7,  3,  8, 11,
        4,  4,  2,  6,  5,  4, 11, 11,  4,  4,  9, 11,  4, 10,  8,  4,  4,
        4,  2, 12,  6,  2,  7,  7,  2,  3, 11, 11,  3, 11,  8,  4,  4,  2,
        6,  7,  4, 10, 11,  4,  2,  5,  5,  2,  7,  5,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert i1m_str == 'mul(truediv(sub(add(add(y, 1), add(3, 2)), add(add(3, 3), mul(4.0, x))), truediv(truediv(sub(2, 1), truediv(x, x)), truediv(truediv(5.0, x), truediv(6.0, 4.0)))), truediv(truediv(truediv(sub(y, 2), sub(3, 3)), sub(mul(x, x), mul(x, 4.0))), truediv(truediv(sub(2, 3), truediv(6.0, x)), truediv(sub(1, 1), sub(3, 1)))))'

def test_tree_point_mutation_st2():
    np.random.seed(42)
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
    pop = pg.make_tree_population(1, pset, 3, 8, 150, init_method=pg.full_tree)
    i1 = pop.individuals[0].clone()
    i1m = pg.tree_point_mutation(i1, pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert i1.depth == i1m.depth
    assert i1.nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([ 3,  2,  1,  2,  3,  6,  6,  3,  4,  6,  2,  1,  5,  4,  3,  6,  5,
        3,  2,  1, 11, 11,  3,  6,  5,  2,  1,  5, 11,  3,  5,  5,  2,  3,
        2,  1, 11,  4,  1, 11,  6,  2,  3,  4, 11,  1,  4,  6,  1,  2,  1,
        4, 11,  3, 11, 11,  2,  1,  5,  5,  1,  4,  6,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert i1m_str == 'mul(sub(add(sub(mul(3, 3), mul(1, 3)), sub(add(2, 1), mul(3, 2))), mul(sub(add(y, y), mul(3, 2)), sub(add(2, y), mul(2, 2)))), sub(mul(sub(add(y, 1), add(y, 3)), sub(mul(1, y), add(1, 3))), add(sub(add(1, y), mul(y, y)), sub(add(2, 2), add(1, 3)))))'
