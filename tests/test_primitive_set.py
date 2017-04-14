import sys
sys.path.append('../')

import numpy as np
import pygenome as pg
import operator as op

def test_primitive_set_add_functions():

    pset = pg.PrimitiveSet()
    pset.addFunction(op.add, 2)
    pset.addFunction(op.add, 2)
    pset.addFunction(op.add, 2)

    assert pset.num_primitives == 3
    assert list(pset.functions.keys()) == [1, 2, 3]
    assert pset.arity_cache == {2: [1, 2, 3]}
