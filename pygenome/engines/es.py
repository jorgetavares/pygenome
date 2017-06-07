import numpy as np
from pygenome.representations.linear import (
    uniform_chromossome
)
from pygenome.representations.population import (
    make_normal_population,
    make_uniform_population,
    make_adaptive_population
)
from pygenome.fitness.evaluation import evaluate_population
from pygenome.operators.crossover import (
    apply_global_crossover,
    intermediary_crossover,
    intermediary_crossover_adaptive
)
from pygenome.operators.mutation import (
    apply_global_mutation,
    uncorrelated_one_step_mutation,
    uncorrelated_one_step_mutation_adaptive
)
from pygenome.operators.selection import best_individual
from pygenome.operators.replacement import mu_comma_lambda_replacement
from pygenome.engines.generic import generic_es

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


def evolutionary_strategy_adaptive(fitness_fn, chr_size, low, high, pop_size=30,
                                   total_generations=100,
                                   mt=uncorrelated_one_step_mutation_adaptive,
                                   cx=intermediary_crossover_adaptive, pool_size=100,
                                   epsilon=1e-08,
                                   replace_pop=mu_comma_lambda_replacement,
                                   make_pop='uniform'):
    if mt == uncorrelated_one_step_mutation_adaptive:
        ext_chr = 1
    else:
        ext_chr = chr_size

    if make_pop == 'uniform':
        def make_pop():
            p = make_uniform_population(pop_size, chr_size, low=low, high=high)
            return make_adaptive_population(p, uniform_chromossome, ext_chr, low=low, high=high)
    else:
        def make_pop():
            p = make_normal_population(pop_size, chr_size, mean=low, sigma=high)
            return make_adaptive_population(p, uniform_chromossome, ext_chr, low=low, high=high)

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