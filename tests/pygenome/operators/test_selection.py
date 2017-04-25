import numpy as np
import pygenome as pg


def test_best_individual():
    np.random.seed(42)
    pop = pg.make_integer_population(10, 5)
    pop = pg.evaluate_population(pop, lambda x : 1. / (1. + pg.onemax(x)))
    best = pg.best_individual(pop)
    assert np.array_equal(best.genotype, np.array([1, 1, 1, 1, 1]))
    assert best.fitness.value == 0.16666666666666666
    
def test_worst_individual():
    np.random.seed(42)
    pop = pg.make_integer_population(10, 5)
    pop = pg.evaluate_population(pop, lambda x : 1. / (1. + pg.onemax(x)))
    worst = pg.worst_individual(pop)
    assert np.array_equal(worst.genotype, np.array([0, 1, 0, 0, 0]))
    assert worst.fitness.value == 0.5
