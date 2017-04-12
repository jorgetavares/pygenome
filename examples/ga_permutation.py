import sys
sys.path.append('../')

import numpy as np
import pygenome as pg

# fitness function: measures the sortness of a permutation
def sorted_permutation(vector):
    unsorted = vector.size
    for i in range(vector.size):
        if vector[i] == i:
            unsorted -= 1
    return unsorted    

permutation_size = 10

# GA 1
def generational_no_elitism():
    np.random.seed(42)
    pop = pg.genetic_algorithm_permutation(sorted_permutation, permutation_size, total_generations=25)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genotype))

# GA 2
def generational_with_elitism():
    np.random.seed(42)
    pop = pg.genetic_algorithm_permutation(sorted_permutation, permutation_size, total_generations=25, elitism=True)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genotype))

# entry point
if __name__ == "__main__":
    print('GA 1: generational, no elitism')
    generational_no_elitism()

    print('GA 2: generational, with elitism')
    generational_with_elitism()    
