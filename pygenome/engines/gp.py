import numpy as np

from pygenome.representations.population import make_tree_population
from pygenome.fitness.evaluation import evaluate_tree_population
from pygenome.utilities.logging import evolution_progress
from pygenome.operators.crossover import (
    apply_crossover,
    tree_crossover
)
from pygenome.operators.mutation import (
    apply_mutation,
    subtree_mutation
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

# standard integer GA
def genetic_programming(fitness_fn, pset, min_depth, max_depth, chr_size, pop_size=100,
                        total_generations=20,
                        cx=tree_crossover, cx_rate=0.7,
                        mt=subtreemutation, ind_mt_rate=1.0, op_mt_rate=0.01,
                        select_fn=tournament_selection,
                        elitism=False, generational=True, initial_type=None, init_method=grow_tree):
   
    def make_pop():
        return make_tree_population(pop_size, pset, min_depth, max_depth, chr_size, initial_type=initial_type, init_method=init_method)

    def eval_pop(p):
        return evaluate_tree_population(p, fitness_fn)

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
        return apply_crossover(p, cx_rate, cx, pset=pset)

    def apply_mt(p):
        return apply_mutation(
            p, ind_mt_rate, mt, gene_rate=op_mt_rate, pset=pset)

    # run ga
    pop = generic_ea(total_generations, make_pop, eval_pop,
                     select_pop, apply_cx, apply_mt, replace_pop)

    return pop
