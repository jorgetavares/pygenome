import numpy as np
import pygenome as pg

def test_generational():
    np.random.seed(42)
    size = 1000
    ind_size = 20
    pop1 = pg.make_integer_population(size, ind_size)
    pop1_clone = pop1.clone()
    pop2 = pg.make_integer_population(size, ind_size)
    pop1 = pg.generational_replacement(pop1, pop2)
    assert pop1.equal(pop2) is True
    assert pop1.equal(pop1_clone) is False


