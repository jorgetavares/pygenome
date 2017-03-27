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

# GA
def generational_with_elitism():
    np.random.seed(42)
    pop = algo.genetic_algorithm(fitness_fn, sphere_model_size, -5.0, 5.0, elitism=True, mt=mutation.uniform_mutation, total_generations=200)
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))


# entry point
if __name__ == "__main__":
    print('GA: generational, with elitism')
    generational_with_elitism()    
 