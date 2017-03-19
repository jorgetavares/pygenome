import numpy as np

import pygenome.fitness as fitness
import pygenome.population as population
import pygenome.selection as selection
import pygenome.crossover as crossover
import pygenome.mutation as mutation
import pygenome.replacement as replacement
import pygenome.logging as logging


# TODO: use only builders here so that other functions bukd on top
def generic_ea():

    # build population
    pop = population.make_integer_population(size, ind_size, low=low, high=high)
    pop = fitness.evaluate_population(pop, fitness_func, fitness_fn_kargs)
    logging.evolution_progress(1, pop)

    # evolutionary loop
    for i in range(2, total_generations):
        selection.select_population(pop, selection_fn, selection_fn_kargs)
        offsprings = crossover.apply_crossover(pop, ...)
        offsprings = mutation.apply_mutation(pop, ...)
        replacement.generational(pop, offsprings)
        fitness.evaluate_population(pop, fitness_func, fitness_fn_kargs)
        logging.evolution_progress(1, pop)

    # end