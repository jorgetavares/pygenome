import numpy as np
import pygenome as pg

def test_select_population():
    np.random.seed(42)
    pop = pg.make_integer_population(10, 5)
    pop = pg.evaluate_population(pop, lambda x : 1. / (1. + pg.onemax(x)))
    diff = 0
    for i in range(pop.size):
        if not np.array_equal(pop.individuals[i].genotype, pop.individuals[i].genotype):
            diff +=1
    assert diff == 0

    original_pop = pop.clone()
    pop = pg.select_population(pop, pg.tournament_selection)
    for i in range(pop.size):
        if not np.array_equal(pop.individuals[i].genotype, original_pop.individuals[i].genotype):
            diff +=1
    assert diff != 0

def test_select_steadystate_population():
    np.random.seed(42)
    pop = pg.make_integer_population(10, 5)
    pop = pg.evaluate_population(pop, lambda x : 1. / (1. + pg.onemax(x)))
    diff = 0
    for i in range(pop.size):
        if not np.array_equal(pop.individuals[i].genotype, pop.individuals[i].genotype):
            diff +=1
    assert diff == 0

    original_pop = pop.clone()
    pop = pg.select_steadystate_population(pop, pg.tournament_selection)
    for i in range(pop.size):
        if not np.array_equal(pop.individuals[i].genotype, original_pop.individuals[i].genotype):
            diff +=1
    assert diff != 0

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

def test_random_selection():
    np.random.seed(42)
    individuals = [pg.Individual(fitness=pg.Fitness(value=np.random.randint(10))) for _ in range(10)]    
    pop = pg.Population(individuals=np.array(individuals))
    # [6, 3, 7, 4, 6, 9, 2, 6, 7, 4]
    np.random.seed(42)
    inds1 = [pg.random_selection(pop) for _ in range(10)]
    assert [inds1[i].fitness.value for i in range(10)] == [2, 4, 6, 6, 2, 4, 7, 2, 6, 6]
    np.random.seed(24)
    inds2 = [pg.random_selection(pop) for _ in range(10)]
    assert [inds2[i].fitness.value for i in range(10)] == [7, 4, 6, 6, 3, 3, 3, 6, 6, 4]
    
def test_tournament_selection():
    np.random.seed(42)
    individuals = [pg.Individual(fitness=pg.Fitness(value=np.random.randint(10))) for _ in range(10)]    
    pop = pg.Population(individuals=np.array(individuals))
    # [6, 3, 7, 4, 6, 9, 2, 6, 7, 4]
    np.random.seed(42)
    inds1 = [pg.tournament_selection(pop, size=5) for _ in range(10)]
    assert [inds1[i].fitness.value for i in range(10)] == [2, 2, 4, 6, 3, 4, 4, 2, 2, 2]
    np.random.seed(24)
    inds2 = [pg.tournament_selection(pop, size=5) for _ in range(10)]
    assert [inds2[i].fitness.value for i in range(10)] == [4, 3, 4, 4, 4, 3, 2, 2, 3, 3]
 
def test_negative_tournament_selection():
    np.random.seed(42)
    individuals = [pg.Individual(fitness=pg.Fitness(value=np.random.randint(10))) for _ in range(10)]    
    pop = pg.Population(individuals=np.array(individuals))
    # [6, 3, 7, 4, 6, 9, 2, 6, 7, 4]
    np.random.seed(42)
    inds1 = [pg.negative_tournament_selection(pop, size=5) for _ in range(10)]
    assert [inds1[i].fitness.value for i in range(10)] == [6, 7, 6, 9, 9, 9, 7, 7, 7, 7]
    np.random.seed(24)
    inds2 = [pg.negative_tournament_selection(pop, size=5) for _ in range(10)]
    assert [inds2[i].fitness.value for i in range(10)] == [7, 6, 7, 7, 6, 6, 3, 4, 6, 7]

def test_roulette_wheel_selection():
    np.random.seed(42)
    individuals = [pg.Individual(fitness=pg.Fitness(value=np.random.randint(10))) for _ in range(10)]    
    pop = pg.Population(individuals=np.array(individuals))
    # [6, 3, 7, 4, 6, 9, 2, 6, 7, 4]
    np.random.seed(42)
    inds1 = [pg.roulette_wheel_selection(pop) for _ in range(10)]
    assert [inds1[i].fitness.value for i in range(10)] == [4, 4, 2, 2, 3, 3, 6, 7, 2, 2]
    np.random.seed(24)
    inds2 = [pg.roulette_wheel_selection(pop) for _ in range(10)]
    assert [inds2[i].fitness.value for i in range(10)] == [4, 2, 4, 3, 4, 6, 4, 4, 3, 4]
 