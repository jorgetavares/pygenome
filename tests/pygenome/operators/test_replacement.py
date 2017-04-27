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

# this tests needs to be improved
def test_steady_state():
    np.random.seed(42)
    size1 = 10
    ind_size = 20
    pop1 = pg.make_integer_population(size1, ind_size)
    pop1_clone = pop1.clone()
    size2 = 2
    pop2 = pg.make_integer_population(size2, ind_size)
    pop1 = pg.steady_state_replacement(pop1, pop2)
    found1 = False
    found2 = False
    for ind in pop1.individuals:
        if ind.equal(pop2.individuals[0]) is True:
            found1 = True
        if ind.equal(pop2.individuals[1]) is True:
            found2 = True
        if found1 and found2:
            break

    assert found1 and found2 is True
