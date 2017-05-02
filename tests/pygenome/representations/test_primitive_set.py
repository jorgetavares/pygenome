import numpy as np
import pygenome as pg
import operator as op

# add non-typed functions
def test_primitive_set_add_functions():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addFunction(op.mul, 2)

    assert pset.typed is False
    assert pset.num_primitives == 3
    assert list(pset.functions.keys()) == [1, 2, 3]
    assert pset.arity_cache == {2: [1, 2, 3]}
    assert pset.functions_types == {}

# add non-typed functions
def test_primitive_set_add_typed_functions():
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [int, int, int])
    pset.addFunction(op.sub, 2, [int, int, int])
    pset.addFunction(op.mul, 2, [float, float, float])

    assert pset.typed is True
    assert pset.num_primitives == 3
    assert list(pset.functions.keys()) == [1, 2, 3]
    assert pset.arity_cache == {2: [1, 2, 3]}
    assert pset.functions_types == {int: [1, 2], float: [3]}
    
# duplicate functions
def test_duplicate_entries():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.add, 2)
    pset.addTerminal(2)
    pset.addTerminal(2)
    pset.addVariable("x")
    pset.addVariable("x")
    
    assert pset.num_primitives == 3
    assert list(pset.functions.keys()) == [1]
    assert list(pset.terminals.keys()) == [2]
    assert list(pset.variables.keys()) == [3]
    assert pset.arity_cache == {2: [1]}
    assert pset.primitives == set([op.add, 2, "x"])

def float_constants():
    return np.random.uniform()

def test_make_ephemeral_constants():
    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.sub, 2)
    pset.addTerminal(float_constants, types=None, ephemeral=True)

    assert pset.num_primitives == 3
    assert list(pset.functions.keys()) == [1, 2]
    assert list(pset.terminals.keys()) == [3]
    assert list(pset.variables.keys()) == []
    assert pset.arity_cache == {2: [1, 2]}
    assert pset.primitives == set([op.add, op.sub, float_constants])
    assert pset.ephemeral_cache == set([3])
    
def test_make_ephemeral_constants_typed():
    pset = pg.PrimitiveSet(typed=True)
    pset.addFunction(op.add, 2, [float, float, float])
    pset.addFunction(op.sub, 2, [float, float, float])
    pset.addTerminal(float_constants, types=[float], ephemeral=True)

    assert pset.num_primitives == 3
    assert list(pset.functions.keys()) == [1, 2]
    assert list(pset.terminals.keys()) == [3]
    assert list(pset.variables.keys()) == []
    assert pset.arity_cache == {2: [1, 2]}
    assert pset.primitives == set([op.add, op.sub, float_constants])
    assert pset.ephemeral_cache == set([3])
    assert pset.functions_types == {float: [1, 2]}
    assert pset.terminals_types == {float: [3]}
