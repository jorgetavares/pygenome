import numpy as np
from copy import deepcopy

class Individual(object):
    '''
    Base class for all type of individuals
    '''

    def __init__(self, fitness=None, genome=None):
        self.fitness = fitness # this should be an object since fitness can be more than a simple value
        self.genome = genome   # this should be an object that can be of any type 

    def clone(self):
        return deepcopy(self)


class TreeIndividual(Individual):
    '''
    Class for GP based individuals
    '''

    def __init__(self, fitness=None, tree=None, depth=None, nodes=None):
        super(Individual, self).__init__(fitness=fitness, genome=tree)
        self.depth = depth
        self.nodes = nodes
