import numpy as np

from pygenome.representations.grammar import Grammar
from pygenome.representations.population import make_integer_population
from pygenome.fitness.evaluation import evaluate_grammar_population
from pygenome.utilities.logging import evolution_progress
from pygenome.operators.crossover import (
    apply_crossover,
    uniform_crossover
)
from pygenome.operators.mutation import (
    apply_mutation,
    flip_mutation
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
    steady_state_replacement
)
from pygenome.engines.ga import generic_ea

# standard GE


def grammatical_evolution(grammar, fitness_fn, chr_size, low, high, pop_size=100,
                          total_generations=20,
                          cx=uniform_crossover, cx_rate=0.7,
                          mt=flip_mutation, ind_mt_rate=1.0, op_mt_rate=0.01,
                          select_fn=tournament_selection,
                          elitism=False, generational=True, wrap=True):
    # config ga components
    make_pop_type = make_integer_population

    def make_pop():
        return make_pop_type(pop_size, chr_size, low=low, high=high)

    def eval_pop(p):
        return evaluate_grammar_population(p, grammar, wrap, fitness_fn)

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
