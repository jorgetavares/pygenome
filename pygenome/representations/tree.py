import numpy as np
import operator as op
import pygenome as pg

# container for the functiona and terminal set
class PrimitiveSet(object):

    def __init__(self, typed=False):
        self.functions = {}
        self.terminals = {}
        self.variables = {}
        self.num_primitives = 0
        self.typed = typed

    def addFunction(self, fn, arity, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')
        primitive = (fn, arity, types)
        self.num_primitives += 1
        self.functions[self.num_primitives] = primitive

    def addTerminal(self, term, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')
        primitive = (term, types)
        self.num_primitives += 1
        self.terminals[self.num_primitives] = primitive

    def addVariable(self, var, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed variable set so types are required!')   
        variable = (var, types)
        self.num_primitives += 1
        self.variables[self.num_primitives] = variable


def interpreter(pset, tree, run=False, vars_inputs=None):
    '''
    interpreter

    Args:
        pset (PrimitiveSet): set of functions, terminals and variables
        tree (array): tree encoded as an array
        run (boolean): runs the program if true, else outputs string 

    Returns:
        execution of the tree or the string representation 
    '''
    def run_tree(array_tree):
        element = array_tree[run_tree.position]
        run_tree.position +=1

        if element in pset.terminals:
            terminal, _ = pset.terminals[element]
            return terminal if run else str(terminal)
        elif element in pset.variables:
            variable, _ = pset.variables[element]    
            return vars_inputs[variable] if run else variable 
        elif element in pset.functions:
            fn, arity, _ = pset.functions[element]
            args = []
            for i in range(arity):
                args.append(run_tree(array_tree))
            return fn(*args) if run else fn.__name__ + "(" + ", ".join(args) + ")" 
        else:
            raise AttributeError('Primitive not found in Primitive Set!')
    
    run_tree.position = 0
    result = run_tree(tree)
    return result
