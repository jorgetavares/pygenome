import sys
sys.path.append('../')

import numpy as np

import pygenome.algorithms as algo
import pygenome.individual as individual
import pygenome.fitness as fitness
import pygenome.selection as selection

#TODO: BUG: verify why elistism is not working

# fitness function: measures the sortness of a permutation
def sorted_permutation(vector):
    unsorted = vector.size
    for i in range(vector.size):
        if vector[i] == i:
            unsorted -= 1
    return unsorted    

permutation_size = 20

# GA 1
def generational_no_elitism():
    np.random.seed(42)
    pop = algo.genetic_algorithm_permutation(sorted_permutation, permutation_size, total_generations=30)
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# GA 2
def generational_with_elitism():
    np.random.seed(42)
    pop = algo.genetic_algorithm_permutation(sorted_permutation, permutation_size, total_generations=30, elitism=True)
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# entry point
if __name__ == "__main__":
    print('GA 1: generational, no elitism')
    generational_no_elitism()

    print('GA 2: generational, with elitism')
    generational_with_elitism()    
