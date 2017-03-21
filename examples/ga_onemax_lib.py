import sys
sys.path.append('../')

import numpy as np

import pygenome.fitness as fitness
import pygenome.population as population
import pygenome.selection as selection
import pygenome.crossover as crossover
import pygenome.mutation as mutation
import pygenome.replacement as replacement
import pygenome.logging as logging


# config GA, minimization fitness function required
fitness_fn = lambda x : 1. / fitness.onemax(x) 
onemax_size = 50

# manually coded GAs to show case library style
def genetic_algorithm_binary(fitness_fn, chr_size, pop_size=100, total_generations=20, 
                             cx_rate=0.7, ind_mt_rate=1.0, op_mt_rate=0.01, elitism=False):
    # initial population
    pop = population.make_integer_population(pop_size, chr_size)
    pop = fitness.evaluate_population(pop, fitness_fn)
    logging.evolution_progress(0, pop)

    # evolutionary loop
    for i in range(1, total_generations):
        pop = selection.select_population(pop, selection.tournament_selection)
        offsprings = crossover.apply_crossover(pop, cx_rate, crossover.uniform_crossover)
        offsprings = mutation.apply_mutation(offsprings, ind_mt_rate, mutation.flip_mutation, gene_rate=op_mt_rate)
        if elitism:
            pop = replacement.elite_strategy(replacement.generational(pop, offsprings), selection.best_individual(pop))
        else:   
            pop = replacement.generational(pop, offsprings)
        pop = fitness.evaluate_population(pop, fitness_fn)
        logging.evolution_progress(i, pop)

    return pop

# GA 1
def generational_no_elitism():
    np.random.seed(42)
    pop = genetic_algorithm_binary(fitness_fn, onemax_size)
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# GA 2
def generational_with_elitism():
    np.random.seed(42)
    pop = genetic_algorithm_binary(fitness_fn, onemax_size, elitism=True)
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# entry point
if __name__ == "__main__":
    print('GA 1: generational, no elitism')
    generational_no_elitism()

    print('GA 2: generational, with elitism')
    generational_with_elitism() 
