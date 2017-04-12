import sys
sys.path.append('../')

import numpy as np
import pygenome as pg

# config GA, minimization fitness function required
fitness_fn = lambda x : 1. / pg.onemax(x) 
onemax_size = 50

# GA 1
def generational_no_elitism():
    np.random.seed(42)
    pop = pg.genetic_algorithm_binary(fitness_fn, onemax_size)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genotype))

# GA 2
def generational_with_elitism():
    np.random.seed(42)
    pop = pg.genetic_algorithm_binary(fitness_fn, onemax_size, elitism=True)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genotype))

# GA 3
def steady_state_no_elitism():
    np.random.seed(42)
    pop = pg.genetic_algorithm_binary(fitness_fn, onemax_size, generational=False, total_generations=200)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genotype))

# GA 4
def steady_state_with_elitism():
    np.random.seed(42)
    pop = pg.genetic_algorithm_binary(fitness_fn, onemax_size, generational=False, total_generations=200, elitism=True)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genotype))

# entry point
if __name__ == "__main__":
    print('GA 1: generational, no elitism')
    generational_no_elitism()

    print('GA 2: generational, with elitism')
    generational_with_elitism()    

    print('GA 3: steady state, no elitism')
    steady_state_no_elitism()   

    print('GA 4: steady state, with elitism')
    steady_state_with_elitism()   