import numpy as np
from copy import deepcopy

class Individual(object):
    '''
    Base class for all type of individuals
    '''

    def __init__(self, fitness=None, genotype=None):
        self.fitness = fitness # this should be an object since fitness can be more than a simple value
        self.genotype = genotype   # this should be an object that can be of any type 
        self.run_eval = True # the individual requires evaluation

    def clone(self):
        return deepcopy(self)

    def equal(self, ind):
        if self.fitness is not None:
            fit = self.fitness.equal(ind.fitness)
        else:
            fit = self.fitness == ind.fitness
        gen = np.array_equal(self.genotype, ind.genotype)
        eva = self.run_eval == ind.run_eval
        return fit and gen and eva


class TreeIndividual(Individual):
    '''
    Class for GP based individuals
    '''

    def __init__(self, fitness=None, tree=None, depth=None, nodes=None):
        super().__init__(fitness=fitness, genotype=tree)
        self.depth = depth
        self.nodes = nodes

    def equal(self, ind):
        depth = self.depth == ind.depth
        nodes = self.nodes == ind.nodes
        return super().equal(ind) and depth and nodes
