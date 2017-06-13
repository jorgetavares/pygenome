import numpy as np
import pygenome as pg

# config GA, minimization fitness function required
# adapted from DEAP:
# https://github.com/DEAP/deap/blob/master/examples/ga/nqueens.py
def evalNQueens(individual):
    """Evaluation function for the n-queens problem.
    The problem is to determine a configuration of n queens
    on a nxn chessboard such that no queen can be taken by
    one another. In this version, each queens is assigned
    to one column, and only one queen can be on each line.
    The evaluation function therefore only counts the number
    of conflicts along the diagonals.
    """
    #Count the number of conflicts with other queens.
    #The conflicts can only be diagonal, count on each diagonal line
    left_diagonal = [0] * (2*individual.size-1)
    right_diagonal = [0] * (2*individual.size-1)
    
    #Sum the number of queens on each diagonal:
    for i in range(individual.size):
        left_diagonal[i+individual[i]] += 1
        right_diagonal[individual.size-1-i+individual[i]] += 1
    
    #Count the number of conflicts on each diagonal
    sum_ = 0
    for i in range(2*individual.size-1):
        if left_diagonal[i] > 1:
            sum_ += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            sum_ += right_diagonal[i] - 1
    return sum_


fitness_fn = lambda x : 1. / evalNQueens(x) 
nqueens_size = 20

# GA 1
def generational_with_elitism():
    np.random.seed(42)
    pop = pg.genetic_algorithm_permutation(fitness_fn, nqueens_size, elitism=True, total_generations=200, cx_rate=0.5, ind_mt_rate=0.2, op_mt_rate=3.0/nqueens_size)
    best = pg.best_individual(pop)
    print('fitness: %s\tgenotype: %s' % (best.fitness.value, best.genotype))

# entry point
if __name__ == "__main__":
    print('GA 1: generational, with elitism')
    generational_with_elitism()    
