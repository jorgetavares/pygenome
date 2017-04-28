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

def test_elite_strategy():
    np.random.seed(42)
    size = 10
    ind_size = 20
    pop = pg.make_integer_population(size, ind_size)
    fitness_function = lambda x : 1. / pg.onemax(x) 
    pop = pg.make_integer_population(size, ind_size)
    pop = pg.evaluate_population(pop, fitness_function)
    best = pg.best_individual(pop)
    pop = pg.elite_strategy(pop, best)
    found = False
    for ind in pop.individuals:
        if ind.equal(best) is True:
            found = True
            break

    assert found is True

def test_m_comma_lambda():
    np.random.seed(42)
    sphere_model_size = 30
    fitness_fn = lambda x : pg.sphere_model(x[:sphere_model_size]) 
    size = 10
    pop1 = pg.make_uniform_population(size, sphere_model_size)
    pop1 = pg.evaluate_population(pop1, fitness_fn)
    pop1_clone = pop1.clone()
    pop2 = pg.make_uniform_population(size, sphere_model_size)
    pop2 = pg.evaluate_population(pop2, fitness_fn)
    pop1 = pg.mu_comma_lambda_replacement(pop1, pop2)
    f_p1c = np.array([pop1_clone.individuals[i].fitness.value for i in range(size)])
    f_p2 = np.array([pop2.individuals[i].fitness.value for i in range(size)])
    f_p1 = np.array([pop1.individuals[i].fitness.value for i in range(size)])
    res = np.array([7.7633264242708808, 8.4249828886076532, 9.4585755842303119, 9.8351682100669695, 10.109666113362849, 10.120464613773503, 11.148338435537882, 12.62638917659477, 12.68750163403401, 13.486951247705489])
    assert np.allclose(f_p1, res) is True
    assert np.allclose(f_p1, f_p1c) is False
    assert np.isclose(f_p1.any(), f_p2.any()) is True

def test_m_plus_lambda():
    np.random.seed(42)
    sphere_model_size = 30
    fitness_fn = lambda x : pg.sphere_model(x[:sphere_model_size]) 
    size = 10
    pop1 = pg.make_uniform_population(size, sphere_model_size)
    pop1 = pg.evaluate_population(pop1, fitness_fn)
    pop1_clone = pop1.clone()
    pop2 = pg.make_uniform_population(size, sphere_model_size)
    pop2 = pg.evaluate_population(pop2, fitness_fn)
    pop1 = pg.mu_plus_lambda_replacement(pop1, pop2)
    f_p1c = np.array([pop1_clone.individuals[i].fitness.value for i in range(size)])
    f_p2 = np.array([pop2.individuals[i].fitness.value for i in range(size)])
    f_p1 = np.array([pop1.individuals[i].fitness.value for i in range(size)])
    res = np.array([ 7.76332642,  8.08389311,  8.42498289,  8.77341594,  8.9523664 , 9.45857558,  9.49688766,  9.52769953,  9.7264757 ,  9.83516821])
    assert np.allclose(f_p1, res) is True
    assert np.allclose(f_p1, f_p1c) is False
    assert np.isclose(f_p1.any(), f_p2.any()) is True
