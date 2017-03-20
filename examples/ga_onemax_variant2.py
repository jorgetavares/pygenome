import sys
sys.path.append('../')

import pygenome.fitness as fitness
import pygenome.population as population
import pygenome.selection as selection
import pygenome.crossover as crossover
import pygenome.mutation as mutation
import pygenome.replacement as replacement
import pygenome.logging as logging


# set seed
np.random.seed(42)

# config GA, minimization fitness function required
fitness_fn = lambda x : 1. / fitness.onemax(x) 
onemax_size = 10

# manually coded GA to show case library style
def genetic_algorithm_binary(fitness_fn, chr_size, pop_size=100, total_generations=10, 
                             cx_rate=0.7, ind_mt_rate=1.0, op_mt_rate=0.01):
    # initial population
    pop = population.make_integer_population(pop_size, chr_size)
    pop = fitness.evaluate_population(pop, fitness_fn)
    logging.evolution_progress(1, pop)

    # evolutionary loop
    for i in range(2, total_generations):
        selection.select_population(pop, selection.tournament_selection)
        offsprings = crossover.apply_crossover(pop, cx_rate, crossover.uniform_crossover)
        offsprings = mutation.apply_mutation(offsprings, ind_mt_rate, mutation.flip_mutation, flip_rate=op_mt_rate)
        pop = replacement.generational(pop, offsprings)
        pop = fitness.evaluate_population(pop, fitness_fn)
        logging.evolution_progress(i, pop)

    # do post-run operations
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))


# run GA
genetic_algorithm_binary(fitness_fn, onemax_size)
