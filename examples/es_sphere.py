import numpy as np
import pygenome as pg

# config ES, minimization fitness function required
sphere_model_size = 30


def fitness_fn(x): return pg.sphere_model(x[:sphere_model_size])


# ES 1
def es_mu_comma_lambda_one_step():
    np.random.seed(42)
    pop = pg.evolutionary_strategy(fitness_fn, sphere_model_size, -5.0, 5.0)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))

# ES 2


def es_mu_plus_lambda_one_step():
    np.random.seed(42)
    pop = pg.evolutionary_strategy(fitness_fn, sphere_model_size, -5.0, 5.0,
                                   replace_pop=pg.mu_plus_lambda_replacement)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))

# ES 3


def es_mu_comma_lambda_n_step():
    np.random.seed(42)
    pop = pg.evolutionary_strategy(fitness_fn, sphere_model_size, -5.0, 5.0,
                                   mt=pg.uncorrelated_n_steps_mutation, pop_size=100, pool_size=500)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))

# ES 4


def es_mu_plus_lambda_n_step():
    np.random.seed(42)
    pop = pg.evolutionary_strategy(fitness_fn, sphere_model_size, -5.0, 5.0,
                                   replace_pop=pg.mu_plus_lambda_replacement,
                                   mt=pg.uncorrelated_n_steps_mutation, pop_size=100, pool_size=500)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))


# entry point
if __name__ == "__main__":
    print('ES 1: mu,lambda, one step mutation')
    es_mu_comma_lambda_one_step()

    print('ES 2: mu+lambda, one step mutation')
    es_mu_plus_lambda_one_step()

    print('ES 3: mu,lambda, n step mutation')
    es_mu_comma_lambda_n_step()

    print('ES 4: mu+lambda, n step mutation')
    es_mu_plus_lambda_n_step()
