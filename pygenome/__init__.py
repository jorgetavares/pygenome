"""
PyGenome - Evolutionary Algorithms in Python
"""

__version__ = 0.01

from pygenome.representations import individual
from pygenome.representations import population

from pygenome.fitness import evaluation

from pygenome.operators import crossover
from pygenome.operators import mutation
from pygenome.operators import selection
from pygenome.operators import replacement

from pygenome.utilities import logging

from pygenome.engines import algorithms

