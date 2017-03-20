import sys
sys.path.append('../')

import pygenome.algorithms as algo
import pygenome.fitness as fitness

# config GA, minimization fitness function required
fitness_fn = lambda x : 1. / fitness.onemax(x) 
onemax_size = 10

# run GA
algo.genetic_algorithm_binary(fitness_fn, onemax_size)

