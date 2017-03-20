import sys
sys.path.append('../')

import numpy as np

import pygenome.algorithms as algo
import pygenome.individual as individual
import pygenome.fitness as fitness
import pygenome.selection as selection

# set seed
np.random.seed(42)

# config GA, minimization fitness function required
fitness_fn = lambda x : 1. / fitness.onemax(x) 
onemax_size = 50

# run GA
def main():
    # GA execution
    pop = algo.genetic_algorithm_binary(fitness_fn, onemax_size)

    # do post-run operations
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))


# entry point
if __name__ == "__main__":
    main()
