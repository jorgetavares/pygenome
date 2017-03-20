import numpy as np

import pygenome.fitness as fitness
import pygenome.population as population
import pygenome.selection as selection
import pygenome.crossover as crossover
import pygenome.mutation as mutation
import pygenome.replacement as replacement
import pygenome.logging as logging

# framework calls

# standard GA
def genetic_algorithm_binary(fitness_fn, chr_size, pop_size=100, total_generations=10, 
                             cx=crossover.uniform_crossover, cx_rate=0.7, 
                             mt=mutation.flip_mutation, ind_mt_rate=1.0, op_mt_rate=0.01, 
                             select_fn=selection.tournament_selection, 
                             elitism=True, generational=True, **kargs):
    # config ga components
    make_pop = lambda : population.make_integer_population(pop_size, chr_size)
    eval_pop = lambda p : fitness.evaluate_population(p, fitness_fn)
    replace = replacement.generational if generational else replacement.steady_state
    replace_pop = lambda p, o : replace(p, o)
    apply_cx = lambda p : crossover.apply_crossover(p, cx_rate, cx)
    apply_mt = lambda p : mutation.apply_mutation(p, ind_mt_rate, mt, flip_rate=op_mt_rate)
    select_pop = lambda p : selection.select_population(p, select_fn)

    # run ga
    pop = generic_ea(total_generations, make_pop, eval_pop, select_pop, apply_cx, apply_mt, replace_pop)

    # do post-run operations
    best = selection.best_individual(pop)
    print('fitness: %s\tgenome: %s' % (best.fitness.value, best.genome))


# TODO: use only builders here so that other functions build on top
def generic_ea(total_generations, make_pop, eval_pop, select_pop, apply_cx, apply_mt, replace_pop):
    # build population
    pop = make_pop()
    pop = eval_pop(pop)
    logging.evolution_progress(1, pop)

    # evolutionary loop
    for i in range(2, total_generations):
        pop = select_pop(pop)
        offsprings = apply_cx(pop)
        offsprings = apply_mt(offsprings)
        pop = replace_pop(pop, offsprings)
        pop = eval_pop(pop)
        logging.evolution_progress(i, pop)
    
    # return final population
    return pop
    