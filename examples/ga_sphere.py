import sys
sys.path.append('../')

import numpy as np
import pygenome as pg

# config ES, minimization fitness function required
sphere_model_size = 30
fitness_fn = pg.sphere_model

# GA
def generational_with_elitism():
    np.random.seed(42)
    pop = pg.genetic_algorithm(fitness_fn, sphere_model_size, -5.0, 5.0, elitism=True, mt=pg.uniform_mutation, total_generations=200)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))

# entry point
if __name__ == "__main__":
    print('GA: generational, with elitism')
    generational_with_elitism()    
 