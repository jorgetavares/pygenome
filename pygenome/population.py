import numpy as np

import pygenome.individual


class Population(object):
    '''
    Base class for a population
    '''

    def __init__(self):
        self.individuals = None

