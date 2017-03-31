import numpy as np
#import pygenome as pg
import operator as op

# container for the functiona and terminal set
class PrimitiveSet(object):

    def __init__(self, typed=False):
        self.functions = {}
        self.terminals = {}
        self.num_primitives = 0
        self.typed = typed


    def addFunction(self, fn, arity, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')

        if self.typed:
            primitive = (fn, arity, types)
        else:
            primitive = (fn, arity)

        self.num_primitives += 1
        self.functions[self.num_primitives] = primitive


    def addTerminal(self, term, type=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')
        
        if self.typed:
            primitive = (term, 0, types)
        else:
            primitive = (term, 0)

        self.num_primitives += 1
        self.terminals[self.num_primitives] = primitive


def run_tree(pset, tree, code=False):

    def _run_tree(array_tree):
        element = array_tree[run_tree.position]
        run_tree.position +=1

        if element in pset.terminals:
            terminal, _ = pset.terminals[element]
            return str(terminal) if code else terminal
        elif element in pset.functions:
            fn, arity = pset.functions[element]
            args = []
            for i in range(arity):
                args.append(_run_tree(array_tree))
            return fn.__name__ + "(" + ", ".join(args) + ")" if code else fn(*args)
        else:
            raise AttributeError('Primitive not found in Primitive Set!')
    
    run_tree.position = 0
    result = _run_tree(tree)
    return result
