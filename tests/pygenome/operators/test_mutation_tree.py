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
    pop = pg.make_tree_population(1, pset, 7, 8, init_method=pg.full_tree)
    i1m = pg.tree_point_mutation(pop.individuals[0], pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert pop.individuals[0].depth == i1m.depth
    assert pop.individuals[0].nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([1, 2, 1, 1, 1, 1, 6, 5, 1, 4, 5, 1, 1, 5, 6, 1, 5, 6, 2, 1, 1, 3, 4,
       2, 6, 4, 1, 2, 6, 5, 1, 3, 6, 2, 1, 1, 1, 5, 3, 1, 5, 3, 2, 2, 4, 6,
       1, 5, 3, 1, 2, 1, 6, 6, 1, 6, 5, 2, 1, 4, 5, 1, 4, 3, 2, 1, 2, 1, 1,
       4, 4, 2, 4, 6, 2, 1, 6, 4, 1, 6, 3, 1, 1, 2, 6, 5, 2, 5, 3, 1, 2, 6,
       4, 1, 4, 6, 2, 2, 2, 2, 5, 4, 2, 3, 6, 2, 2, 3, 4, 2, 6, 5, 2, 1, 1,
       6, 4, 2, 6, 3, 2, 2, 6, 5, 1, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0]))
    assert i1m_str == 'add(sub(add(add(add(add(x, 3), add(2, 3)), add(add(3, x), add(3, x))), sub(add(add(1, 2), sub(x, 2)), add(sub(x, 3), add(1, x)))), sub(add(add(add(3, 1), add(3, 1)), sub(sub(2, x), add(3, 1))), add(sub(add(x, x), add(x, 3)), sub(add(2, 3), add(2, 1))))), sub(add(sub(add(add(2, 2), sub(2, x)), sub(add(x, 2), add(x, 1))), add(add(sub(x, 3), sub(3, 1)), add(sub(x, 2), add(2, x)))), sub(sub(sub(sub(3, 2), sub(1, x)), sub(sub(1, 2), sub(x, 3))), sub(add(add(x, 2), sub(x, 1)), sub(sub(x, 3), add(x, 3))))))'

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
    pop = pg.make_tree_population(1, pset, 7, 8, init_method=pg.full_tree)
    i1m = pg.tree_point_mutation(pop.individuals[0], pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)    
    
    assert pop.individuals[0].depth == i1m.depth
    assert pop.individuals[0].nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([ 3,  4,  2,  1,  1,  3,  8, 10,  1,  6,  7,  1,  1,  5, 12,  1, 12,
       12,  3,  2,  3,  8, 10,  3, 11,  9,  4,  4, 11, 11,  2,  6, 12,  4,
        4,  2,  3,  8, 10,  1,  7,  7,  4,  4, 11, 11,  4, 10,  9,  4,  2,
        3, 10, 11,  1, 12, 12,  2,  1,  5,  7,  1,  7,  7,  2,  3,  4,  2,
        3,  9,  9,  3,  8,  9,  2,  3, 11, 11,  1, 12,  6,  4,  2,  3, 10,
       10,  3,  8, 11,  4,  4, 11,  9,  4,  9, 10,  3,  4,  4,  2,  7,  7,
        4, 11,  8,  4,  4, 11,  9,  2, 12,  5,  4,  4,  2, 12,  7,  4, 11,
        8,  4,  2,  6,  7,  2, 12,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0]))
    assert i1m_str == 'mul(truediv(sub(add(add(mul(4.0, 6.0), add(2, 3)), add(add(1, y), add(y, y))), mul(sub(mul(4.0, 6.0), mul(x, 5.0)), truediv(truediv(x, x), sub(2, y)))), truediv(truediv(sub(mul(4.0, 6.0), add(3, 3)), truediv(truediv(x, x), truediv(6.0, 5.0))), truediv(sub(mul(6.0, x), add(y, y)), sub(add(1, 3), add(3, 3))))), sub(mul(truediv(sub(mul(5.0, 5.0), mul(4.0, 5.0)), sub(mul(x, x), add(y, 2))), truediv(sub(mul(6.0, 6.0), mul(4.0, x)), truediv(truediv(x, 5.0), truediv(5.0, 6.0)))), mul(truediv(truediv(sub(3, 3), truediv(x, 4.0)), truediv(truediv(x, 5.0), sub(y, 1))), truediv(truediv(sub(y, 3), truediv(x, 4.0)), truediv(sub(2, 3), sub(y, 2))))))'

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
    pop = pg.make_tree_population(1, pset, 7, 8, init_method=pg.full_tree)
    i1m = pg.tree_point_mutation(pop.individuals[0], pset=pset, gene_rate=0.5)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert pop.individuals[0].depth == i1m.depth
    assert pop.individuals[0].nodes == i1m.nodes
    assert np.array_equal(i1m.genotype, np.array([ 1,  2,  3,  2,  1,  2,  9,  9,  2, 10,  7,  1,  2,  9,  8,  2,  9,
        9,  2,  1,  2,  9, 10,  2,  7, 10,  3,  2, 10,  9,  2,  8,  7,  1,
        2,  3,  2, 10,  9,  2,  9,  8,  1,  2, 10,  7,  2,  7, 10,  2,  3,
        2,  7,  9,  2, 10,  7,  1,  2,  9,  9,  2,  8,  8,  2,  3,  2,  3,
        2, 10, 10,  2,  9,  9,  3,  2,  8,  8,  2,  9, 10,  2,  3,  2, 10,
       10,  2,  7,  8,  1,  2,  9,  8,  2,  7,  7,  3,  2,  3,  2, 10,  7,
        2, 10,  8,  3,  2,  7,  9,  2,  9,  8,  2,  3,  2, 10, 10,  2,  9,
       10,  1,  2, 10,  7,  2, 10,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0]))
    assert i1m_str == 'add(sub(mul(sub(add(sub(6.0, 6.0), sub(x, 4.0)), add(sub(6.0, 5.0), sub(6.0, 6.0))), sub(add(sub(6.0, x), sub(4.0, x)), mul(sub(x, 6.0), sub(5.0, 4.0)))), add(sub(mul(sub(x, 6.0), sub(6.0, 5.0)), add(sub(x, 4.0), sub(4.0, x))), sub(mul(sub(4.0, 6.0), sub(x, 4.0)), add(sub(6.0, 6.0), sub(5.0, 5.0))))), sub(mul(sub(mul(sub(x, x), sub(6.0, 6.0)), mul(sub(5.0, 5.0), sub(6.0, x))), sub(mul(sub(x, x), sub(4.0, 5.0)), add(sub(6.0, 5.0), sub(4.0, 4.0)))), mul(sub(mul(sub(x, 4.0), sub(x, 5.0)), mul(sub(4.0, 6.0), sub(6.0, 5.0))), sub(mul(sub(x, x), sub(6.0, x)), add(sub(x, 4.0), sub(x, 6.0))))))'

def test_subtree_mutation():
    np.random.seed(42)
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    pset.addTerminal(3)
    pset.addVariable("x")
    pop = pg.make_tree_population(1, pset, 7, 8, init_method=pg.full_tree)
    i1m = pg.subtree_mutation(pop.individuals[0], pset=pset)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert i1m.depth == 7
    assert i1m.nodes == 127
    assert np.array_equal(i1m.genotype, np.array([1, 2, 1, 1, 1, 2, 3, 3, 1, 4, 5, 1, 1, 5, 6, 1, 6, 6, 2, 1, 2, 3, 4,
       2, 6, 4, 2, 2, 6, 6, 1, 3, 6, 2, 2, 1, 2, 3, 3, 1, 5, 5, 2, 2, 6, 6,
       2, 5, 4, 2, 1, 2, 5, 6, 1, 6, 6, 1, 1, 3, 5, 1, 3, 3, 1, 2, 2, 1, 2,
       4, 4, 2, 3, 4, 1, 2, 6, 6, 1, 6, 5, 2, 1, 2, 5, 5, 2, 3, 4, 2, 2, 6,
       4, 2, 4, 4, 2, 2, 2, 1, 5, 4, 2, 6, 4, 2, 2, 6, 4, 1, 6, 5, 2, 2, 1,
       6, 3, 2, 6, 3, 2, 1, 4, 5, 1, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0]))
    assert i1m_str == 'add(sub(add(add(add(sub(1, 1), add(2, 3)), add(add(3, x), add(x, x))), sub(add(sub(1, 2), sub(x, 2)), sub(sub(x, x), add(1, x)))), sub(sub(add(sub(1, 1), add(3, 3)), sub(sub(x, x), sub(3, 2))), sub(add(sub(3, x), add(x, x)), add(add(1, 3), add(1, 1))))), add(sub(sub(add(sub(2, 2), sub(1, 2)), add(sub(x, x), add(x, 3))), sub(add(sub(3, 3), sub(1, 2)), sub(sub(x, 2), sub(2, 2)))), sub(sub(sub(add(3, 2), sub(x, 2)), sub(sub(x, 2), add(x, 3))), sub(sub(add(x, 1), sub(x, 1)), sub(add(2, 3), add(x, 2))))))'

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
    pop = pg.make_tree_population(1, pset, 4, 7, init_method=pg.full_tree)
    i1m = pg.subtree_mutation(pop.individuals[0], pset=pset)
    i1m_str = pg.interpreter(pset, i1m.genotype)

    assert i1m.depth == 4
    assert i1m.nodes == 15
    assert np.array_equal(i1m.genotype, np.array([ 3,  4,  2,  7,  6,  4,  8,  8,  2,  3, 10, 10,  1,  7, 12,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0]))
    assert i1m_str == 'mul(truediv(sub(3, 2), truediv(4.0, 4.0)), sub(mul(6.0, 6.0), add(3, y)))'
