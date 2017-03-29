import numpy as np

from pygenome.representations.population import (
    make_integer_population,
    make_normal_population,
    make_uniform_population
)
from pygenome.fitness.evaluation import evaluate_population
from pygenome.utilities.logging import evolution_progress
from pygenome.operators.crossover import (
    apply_crossover,
    apply_global_crossover,
    uniform_crossover,
    partially_match_crossover,
    intermediary_crossover
)
from pygenome.operators.mutation import (
    apply_mutation,
    apply_global_mutation,
    flip_mutation,
    swap_mutation,
    uncorrelated_one_step_mutation
)
from pygenome.operators.selection import (
    best_individual,
    tournament_selection,
    select_population,
    select_steadystate_population
)
from pygenome.operators.replacement import (
    elite_strategy,
    generational_replacement,
    steady_state_replacement,
    mu_comma_lambda_replacement,
    mu_plus_lambda_replacement
)


# standard binary GA
def genetic_algorithm_binary(fitness_fn, chr_size, pop_size=100,
                             total_generations=20,
                             cx=uniform_crossover, cx_rate=0.7,
                             mt=flip_mutation, ind_mt_rate=1.0,
                             op_mt_rate=0.01,
                             select_fn=tournament_selection,
                             elitism=False, generational=True):
    # run ga
    pop = genetic_algorithm(fitness_fn, chr_size, 0, 1, pop_size=pop_size,
                            total_generations=total_generations,
                            cx=cx, cx_rate=cx_rate, mt=mt,
                            ind_mt_rate=ind_mt_rate,
                            op_mt_rate=op_mt_rate,
                            select_fn=select_fn, elitism=elitism,
                            generational=generational)
    return pop


# standard permutation GA
def genetic_algorithm_permutation(fitness_fn, chr_size, pop_size=100,
                                  total_generations=20,
                                  cx=partially_match_crossover, cx_rate=0.7,
                                  mt=swap_mutation, ind_mt_rate=1.0,
                                  op_mt_rate=None,
                                  select_fn=tournament_selection,
                                  elitism=False, generational=True):
    # run ga
    pop = genetic_algorithm(fitness_fn, chr_size, None, None,
                            pop_size=pop_size,
                            total_generations=total_generations,
                            cx=cx, cx_rate=cx_rate, mt=mt,
                            ind_mt_rate=ind_mt_rate,
                            op_mt_rate=op_mt_rate,
                            select_fn=select_fn, elitism=elitism,
                            generational=generational)
    return pop


# standard integer GA
def genetic_algorithm(fitness_fn, chr_size, low, high, pop_size=100,
                      total_generations=20,
                      cx=uniform_crossover, cx_rate=0.7,
                      mt=flip_mutation, ind_mt_rate=1.0, op_mt_rate=0.01,
                      select_fn=tournament_selection,
                      elitism=False, generational=True):
    # config ga components
    if type(low) is float and type(high) is float:
        make_pop_type = make_uniform_population
    else:
        make_pop_type = make_integer_population

    def make_pop():
        return make_pop_type(pop_size, chr_size, low=low, high=high)

    def eval_pop(p):
        return evaluate_population(p, fitness_fn)

    # config selection/replacement strategy
    if generational:
        select = select_population
        replace = generational_replacement
    else:
        select = select_steadystate_population
        replace = steady_state_replacement

    def select_pop(p): return select(p, select_fn)

    if elitism:
        def replace_pop(p, o):
            b = best_individual(p)
            p2 = replace(p, o)
            return elite_strategy(p2, b)
    else:
        def replace_pop(p, o):
            return replace(p, o)

    # operators and selection
    def apply_cx(p):
        return apply_crossover(p, cx_rate, cx)

    def apply_mt(p):
        return apply_mutation(
            p, ind_mt_rate, mt, gene_rate=op_mt_rate, low=low, high=high)

    # run ga
    pop = generic_ea(total_generations, make_pop, eval_pop,
                     select_pop, apply_cx, apply_mt, replace_pop)

    return pop


# generic EA framework
def generic_ea(total_generations, make_pop, eval_pop,
               select_pop, apply_cx, apply_mt, replace_pop):
    # build population
    pop = make_pop()
    pop = eval_pop(pop)
    evolution_progress(0, pop)

    # evolutionary loop
    for i in range(1, total_generations):
        parents = select_pop(pop)
        offsprings = apply_cx(parents)
        offsprings = apply_mt(offsprings)
        pop = replace_pop(pop, offsprings)
        pop = eval_pop(pop)
        evolution_progress(i, pop)

    # return final population
    return pop


def evolutionary_strategy(fitness_fn, chr_size, low, high, pop_size=30,
                          total_generations=100,
                          mt=uncorrelated_one_step_mutation,
                          cx=intermediary_crossover, pool_size=100,
                          epsilon=1e-08,
                          replace_pop=mu_comma_lambda_replacement,
                          make_pop='uniform'):
    # chromossome size is extended to have self-adaptive mutation rates
    # according to the chosen operator: one_step (+1) or n_steps (*2)
    if mt == uncorrelated_one_step_mutation:
        ext_chr = chr_size + 1
    else:
        ext_chr = chr_size * 2

    if make_pop == 'uniform':
        def make_pop():
            return make_uniform_population(
                pop_size, ext_chr, low=low, high=high)
    else:
        def make_pop():
            return make_normal_population(
                pop_size, ext_chr, mean=low, sigma=high)

    def eval_pop(p):
        return evaluate_population(p, fitness_fn)

    # operators and selection
    def apply_mt(p):
        return apply_global_mutation(p, pool_size, mt, epsilon=epsilon)

    def apply_cx(p):
        return apply_global_crossover(p, cx)

    # run es
    pop = generic_es(total_generations, make_pop,
                     eval_pop, apply_mt, apply_cx, replace_pop)

    return pop


# generic ES framework
def generic_es(total_generations, make_pop, eval_pop,
               apply_mt, apply_cx, replace_pop):
    # build population
    pop = make_pop()
    pop = eval_pop(pop)
    evolution_progress(0, pop)

    # evolutionary loop
    for i in range(1, total_generations):
        offsprings = apply_mt(pop)
        offsprings = apply_cx(offsprings)
        offsprings = eval_pop(offsprings)
        pop = replace_pop(pop, offsprings)
        evolution_progress(i, pop)

    # return final population
    return pop
