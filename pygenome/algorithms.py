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


# generic EA framework
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


def evolutionary_strategy(fitness_fn, chr_size, low, high, pop_size=30, total_generations=100, 
                          mt=mutation.uncorrelated_one_step_mutation,  
                          cx=crossover.intermediary_crossover, pool_size=100, epsilon=0.000000001, 
                          replace_pop=replacement.mu_comma_lambda_replacement,
                          make_pop='uniform'):
    
    # chromossome size is extended to have self-adaptive mutation rates
    # according to the chosen operator: one_step (+1) or n_steps (*2)
    ext_chr = chr_size + 1 if mt == mutation.uncorrelated_one_step_mutation else chr_size * 2

    if make_pop == 'uniform':
        make_pop = lambda : population.make_uniform_population(pop_size, ext_chr, low=low, high=high) 
    else:
        make_pop = lambda : population.make_normal_population(pop_size, ext_chr, mean=low, sigma=high)
    
    eval_pop = lambda p : fitness.evaluate_population(p, fitness_fn)
    
    # operators and selection
    apply_mt = lambda p : mutation.apply_global_mutation(p, pool_size, mt, epsilon=epsilon)
    apply_cx = lambda p : crossover.apply_global_crossover(p, cx)
    
    # run es
    pop = generic_es(total_generations, make_pop, eval_pop, apply_mt, apply_cx, replace_pop)

    return pop    
   

# generic ES framework
def generic_es(total_generations, make_pop, eval_pop, apply_mt, apply_cx, replace_pop):
    # build population
    pop = make_pop()
    pop = eval_pop(pop)
    logging.evolution_progress(0, pop)

    # evolutionary loop
    for i in range(1, total_generations):
        offsprings = apply_mt(pop)
        offsprings = apply_cx(offsprings)
        offsprings = eval_pop(offsprings)
        pop = replace_pop(pop, offsprings)
        logging.evolution_progress(i, pop)
    
    # return final population
    return pop
    