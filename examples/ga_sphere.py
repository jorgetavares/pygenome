import sys
sys.path.append('../')

import numpy as np

import pygenome.algorithms as algo
import pygenome.individual as individual
import pygenome.fitness as fitness
import pygenome.selection as selection
import pygenome.replacement as replacement
import pygenome.mutation as mutation


# config ES, minimization fitness function required
sphere_model_size = 30
fitness_fn = fitness.sphere_model


# GA 1
def generational_no_elitism():
    np.random.seed(42)
    pop = algo.genetic_algorithm_binary(fitness_fn, sphere_model_size, mt=mutation.uniform_mutation)
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# GA 2
def generational_with_elitism():
    np.random.seed(42)
    pop = algo.genetic_algorithm_binary(fitness_fn, sphere_model_size, elitism=True, mt=mutation.uniform_mutation)
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))


# entry point
if __name__ == "__main__":
    print('GA 1: generational, no elitism')
    generational_no_elitism()

    print('GA 2: generational, with elitism')
    generational_with_elitism()    
 