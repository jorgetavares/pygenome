import numpy as np

import pygenome.fitness as fitness
import pygenome.population as population
import pygenome.selection as selection
import pygenome.crossover as crossover
import pygenome.mutation as mutation
import pygenome.replacement as replacement
import pygenome.logging as logging

# framework calls

# standard binary GA
def genetic_algorithm_binary(fitness_fn, chr_size, pop_size=100, total_generations=20, 
                             cx=crossover.uniform_crossover, cx_rate=0.7, 
                             mt=mutation.flip_mutation, ind_mt_rate=1.0, op_mt_rate=0.01, 
                             select_fn=selection.tournament_selection, 
                             elitism=False, generational=True):
    # run ga
    pop = genetic_algorithm(fitness_fn, chr_size, 0, 1, pop_size=pop_size, total_generations=total_generations,
                            cx=cx, cx_rate=cx_rate, mt=mt, ind_mt_rate=ind_mt_rate, op_mt_rate=op_mt_rate,
                            select_fn=select_fn, elitism=elitism, generational=generational)
    return pop

# standard permutation GA
def genetic_algorithm_permutation(fitness_fn, chr_size, pop_size=100, total_generations=20, 
                                  cx=crossover.partially_match_crossover, cx_rate=0.7, 
                                  mt=mutation.swap_mutation, ind_mt_rate=1.0, op_mt_rate=None, 
                                  select_fn=selection.tournament_selection, 
                                  elitism=False, generational=True):
    # run ga
    pop = genetic_algorithm(fitness_fn, chr_size, None, None, pop_size=pop_size, total_generations=total_generations,
                            cx=cx, cx_rate=cx_rate, mt=mt, ind_mt_rate=ind_mt_rate, op_mt_rate=op_mt_rate,
                            select_fn=select_fn, elitism=elitism, generational=generational)
    return pop

# standard integer GA
def genetic_algorithm(fitness_fn, chr_size, low, high, pop_size=100, total_generations=20, 
                      cx=crossover.uniform_crossover, cx_rate=0.7, 
                      mt=mutation.flip_mutation, ind_mt_rate=1.0, op_mt_rate=0.01, 
                      select_fn=selection.tournament_selection, 
                      elitism=False, generational=True):
    # config ga components
    make_pop = lambda : population.make_integer_population(pop_size, chr_size, low=low, high=high)
    eval_pop = lambda p : fitness.evaluate_population(p, fitness_fn)
    
    # config selection/replacement strategy
    replace = replacement.generational if generational else replacement.steady_state
    select = selection.select_population if generational else selection.select_steadystate_population
    select_pop = lambda p : select(p, select_fn)

    if elitism:
        def replace_pop(p, o):
           b = selection.best_individual(p)
           p2 = replace(p, o)
           return replacement.elite_strategy(p2, b)
    else:
        def replace_pop(p, o):
           return replace(p, o)

    # operators and selection
    apply_cx = lambda p : crossover.apply_crossover(p, cx_rate, cx)
    apply_mt = lambda p : mutation.apply_mutation(p, ind_mt_rate, mt, gene_rate=op_mt_rate, low=low, high=high)
    
    # run ga
    pop = generic_ea(total_generations, make_pop, eval_pop, select_pop, apply_cx, apply_mt, replace_pop)

    return pop

# generic EA franework
# TODO: use only builders here so that other functions build on top
def generic_ea(total_generations, make_pop, eval_pop, select_pop, apply_cx, apply_mt, replace_pop):
    # build population
    pop = make_pop()
    pop = eval_pop(pop)
    logging.evolution_progress(0, pop)

    # evolutionary loop
    for i in range(1, total_generations):
        parents = select_pop(pop)
        offsprings = apply_cx(parents)
        offsprings = apply_mt(offsprings)
        pop = replace_pop(pop, offsprings)
        pop = eval_pop(pop)
        logging.evolution_progress(i, pop)
    
    # return final population
    return pop
    