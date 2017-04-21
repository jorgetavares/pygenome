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
    pop = pg.make_tree_population(1, pset, 2, 7, 8, init_method=pg.full_tree)
    i1m = pg.tree_point_mutation(pop.individuals[0], pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert pop.individuals[0].depth == i1m.depth
    assert pop.individuals[0].nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([2, 2, 1, 1, 1, 6, 5, 1, 5, 3, 2, 1, 4, 5, 1, 3, 4, 2, 2, 1, 5, 3, 2,
       6, 6, 2, 2, 4, 6, 1, 3, 5, 1, 2, 1, 1, 6, 3, 1, 5, 5, 1, 1, 5, 6, 2,
       6, 5, 1, 2, 1, 4, 4, 1, 5, 6, 1, 1, 3, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert i1m_str == 'sub(sub(add(add(add(x, 3), add(3, 1)), sub(add(2, 3), add(1, 2))), sub(sub(add(3, 1), sub(x, x)), sub(sub(2, x), add(1, 3)))), add(sub(add(add(x, 1), add(3, 3)), add(add(3, x), sub(x, 3))), add(sub(add(2, 2), add(3, x)), add(add(1, 1), add(1, 1)))))'
    
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
    pop = pg.make_tree_population(1, pset, 3, 7, 8, init_method=pg.full_tree)
    i1m = pg.tree_point_mutation(pop.individuals[0], pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)    
    
    assert pop.individuals[0].depth == i1m.depth
    assert pop.individuals[0].nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([ 3,  4,  2,  1,  7, 12,  1,  5,  6,  4,  2,  7,  7,  2, 12,  7,  4,
        4,  4,  9, 10,  2,  6, 12,  4,  4,  9,  9,  4, 11,  8,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert i1m_str == 'mul(truediv(sub(add(3, y), add(1, 2)), truediv(sub(3, 3), sub(y, 3))), truediv(truediv(truediv(5.0, 6.0), sub(2, y)), truediv(truediv(5.0, 5.0), truediv(x, 4.0))))'

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
    pop = pg.make_tree_population(1, pset, 3, 7, 8, init_method=pg.full_tree)
    i1m = pg.tree_point_mutation(pop.individuals[0], pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert pop.individuals[0].depth == i1m.depth
    assert pop.individuals[0].nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([ 3,  2,  3,  2,  7,  9,  2,  9, 10,  3,  2,  8,  9,  2,  8,  9,  2,
        1,  2,  7,  9,  2, 10,  9,  3,  2, 10, 10,  2,  8,  8,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert i1m_str == 'mul(sub(mul(sub(4.0, 6.0), sub(6.0, x)), mul(sub(5.0, 6.0), sub(5.0, 6.0))), sub(add(sub(4.0, 6.0), sub(x, 6.0)), mul(sub(x, x), sub(5.0, 5.0))))'

def test_subtree_mutation():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    pop = pg.make_tree_population(1, pset, 2, 7, 8, init_method=pg.full_tree)
    i1m = pg.subtree_mutation(pop.individuals[0], pset=pset)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert i1m.depth == 8
    assert i1m.nodes == 67
    assert np.array_equal(i1m.genotype, np.array([1, 2, 1, 1, 1, 6, 3, 1, 2, 4, 1, 4, 6, 4, 1, 1, 5, 5, 2, 3, 6, 2, 2,
       1, 4, 3, 2, 6, 6, 2, 2, 4, 6, 2, 3, 3, 2, 2, 2, 1, 6, 3, 1, 5, 5, 1,
       2, 6, 6, 2, 6, 5, 2, 2, 1, 4, 5, 2, 5, 6, 2, 1, 5, 3, 1, 5, 3, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert i1m_str == 'add(sub(add(add(add(x, 1), add(sub(2, add(2, x)), 2)), add(add(3, 3), sub(1, x))), sub(sub(add(2, 1), sub(x, x)), sub(sub(2, x), sub(1, 1)))), sub(sub(sub(add(x, 1), add(3, 3)), add(sub(x, x), sub(x, 3))), sub(sub(add(2, 3), sub(3, x)), sub(add(3, 1), add(3, 1)))))'

def test_subtree_mutation_typed():
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
    pop = pg.make_tree_population(1, pset, 2, 4, 7, init_method=pg.full_tree)
    i1m = pg.subtree_mutation(pop.individuals[0], pset=pset)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert i1m.depth == 3
    assert i1m.nodes == 7
    assert np.array_equal(i1m.genotype, np.array([ 3,  4,  8, 10,  4,  8,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert i1m_str == 'mul(truediv(4.0, 6.0), truediv(4.0, 4.0))'
