import numpy as np

from pygenome.utilities.logging import evolution_progress

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
