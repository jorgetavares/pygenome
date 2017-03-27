import sys
sys.path.append('../')

import numpy as np
import pygenome as pg

# config GA, minimization fitness function required
fitness_fn = lambda x : 1. / pg.onemax(x) 
onemax_size = 50

# manually coded GAs to show case library style
def genetic_algorithm_binary(fitness_fn, chr_size, pop_size=100, total_generations=20, 
                             cx_rate=0.7, ind_mt_rate=1.0, op_mt_rate=0.01, elitism=False):
    # initial population
    pop = pg.make_integer_population(pop_size, chr_size)
    pop = pg.evaluate_population(pop, fitness_fn)
    pg.evolution_progress(0, pop)

    # evolutionary loop
    for i in range(1, total_generations):
        pop = pg.select_population(pop, pg.tournament_selection)
        offsprings = pg.apply_crossover(pop, cx_rate, pg.uniform_crossover)
        offsprings = pg.apply_mutation(offsprings, ind_mt_rate, pg.flip_mutation, gene_rate=op_mt_rate)
        if elitism:
            pop = pg.elite_strategy(pg.generational(pop, offsprings), pg.best_individual(pop))
        else:   
            pop = pg.generational(pop, offsprings)
        pop = pg.evaluate_population(pop, fitness_fn)
        pg.evolution_progress(i, pop)

    return pop

# GA 1
def generational_no_elitism():
    np.random.seed(42)
    pop = genetic_algorithm_binary(fitness_fn, onemax_size)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# GA 2
def generational_with_elitism():
    np.random.seed(42)
    pop = genetic_algorithm_binary(fitness_fn, onemax_size, elitism=True)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))

# entry point
if __name__ == "__main__":
    print('GA 1: generational, no elitism')
    generational_no_elitism()

    print('GA 2: generational, with elitism')
    generational_with_elitism() 
