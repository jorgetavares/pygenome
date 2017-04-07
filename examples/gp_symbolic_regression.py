import sys
sys.path.append('../')

import numpy as np
import pygenome as pg

# define problem
sphere_model_size = 30
fitness_fn = pg.sphere_model

# GP
def gp_with_elitism():
    np.random.seed(42)
    pop = pg.genetic_programming(fitness_fn, pset, min_depth, max_depth. max_size, elitism=True, total_generations=20)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# entry point
if __name__ == "__main__":
    print('GP: with elitism')
    gp_with_elitism()    
 