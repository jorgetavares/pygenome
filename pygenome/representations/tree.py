import numpy as np
import pygenome as pg
import operators as ops

# container for the functiona and terminal set
class PrimitiveSet(object):

    def __init__(self, typed=False):
        self.functions = None
        self.terminals = None
        self.typed = typed


    def add_function(self, fn, arity, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')

        if typed:
            primitive = (fn, arity, types)
        else:
            primitive = (fn, arity)

        self.functions.append(primitive) 


    def add_terminal(self, term):
        self.terminals.append(term)


# TODO:
# create function + terminal sets

# create tree initializers

# -> operators

# -> eval

# -> example

