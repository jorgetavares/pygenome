import numpy as np
import pygenome as pg
import operator as op

# container for the functiona and terminal set
class PrimitiveSet(object):

    def __init__(self, typed=False):
        self.functions = None
        self.terminals = None
        self.num_primitives = 0
        self.typed = typed


    def addFunction(self, fn, arity, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')

        if typed:
            primitive = (fn, arity, types)
        else:
            primitive = (fn, arity)

        self.num_primitives += 1
        self.functions[self.num_primitives] = primitive


    def addTerminal(self, term, type=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')
        
        if typed:
            primitive = (term, 0, types)
        else:
            primitive = (term, 0)

        self.num_primitives += 1
        self.terminals[num_primitives] = primitive


def run-linear-tree(pset, tree):

    position = 0

    def run-tree(array_tree):
        element = array_tree[position]
        position +=1

        if element in pset.terminals:
            terminal, _ = pset.terminals[element]
            return terminal
        elif element in pset.functions:
            fn, arity = pset.functions[element]
            args = []
            for i in range(arity):
                args.append(run-tree(array_tree))
            return getattr(fn)(*args)
        else:
            raise AttributeError('Primitive not found in Primitive Set!')

    run-tree(tree)


# TODO:
# create function + terminal sets

# create tree initializers

# -> operators

# -> eval

# -> example

