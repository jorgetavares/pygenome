import sys
sys.path.append('../')

import numpy as np
import pygenome as pg


# define problem
# y = f(x) = (x + 1) * (x - 3)
def fn(x):
    return (x + 1) * (x - 3)


def make_single_regression(grammar, fn, fitness_cases, loss=mean_squared_error):

    x_points = np.asarray([x for x in range(num_fitness_cases)])
    y_points = np.asarray([fn(x) for x in x_points])

    def regression(solution):
        x_evals = np.empty(num_fitness_cases)
        for i in range(num_fitness_cases):
            x = x_points[i]
            x_evals[i] = exec(solution)

        return loss(x_evals, y_points)

    return regression


def ge_with_elitism():
    np.random.seed(42)

    ind_size = 100
    low = 0
    high = 255
    wrap = True

    grammar = Grammar(filename='ge_symbolic_regression_grammar.txt')
    num_fitness_cases = 10
    fitness_fn = make_single_regression(grammar, fn, num_fitness_cases)

    pop = pg.grammatical_evolution(
        grammar, fitness_fn, ind_size, low, high, elitism=True, total_generations=20, pop_size=100)

    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))


# entry point
if __name__ == "__main__":
    print('GE: with elitism')
    ge_with_elitism()
