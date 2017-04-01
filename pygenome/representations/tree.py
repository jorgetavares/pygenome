import numpy as np
import operator as op
#import pygenome as pg

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


def grow_tree(pset, max_depth, max_size):
    '''
    Grow Tree 
        
    Args:
        pset (PrimitivSet): set of functions, terminals and variables
        max_depth (int): initialization max depth
        max_size (int): the max size of the array that contains the tree
    Returns:
        random tree in an array using the grow method
    '''
    functions_idx = np.array(list(pset.functions.keys()))
    terminals_idx = np.array(list(pset.terminals.keys()))
    variables_idx = np.array(list(pset.variables.keys()))
    all_terminals_idx = np.concatenate([terminals_idx, variables_idx])
    all_primitives_idx = np.concatenate([functions_idx, all_terminals_idx])

    def grow(depth):
        if depth == 0:
            # return a terminal/variable since it's maximum tree depth
            idx = all_terminals_idx[np.random.randint(all_terminals_idx.size)]
            grow.tree[grow.position] = idx
        else:
            # return a function or a terminal/variable
            idx = all_primitives_idx[np.random.randint(all_primitives_idx.size)]
            if idx in pset.terminals or idx in pset.variables:
                grow.tree[grow.position] = idx
            else:
                fn, arity, types = pset.functions[idx]
                grow.tree[grow.position] = idx
                depth -= 1
                for a in range(arity):
                    grow.position += 1
                    grow(depth)
        
    grow.position = 0
    grow.tree = np.zeros(max_size, dtype=np.int64)
    grow(max_depth)

    return grow.tree


def full_tree(pset, max_depth, max_size):
    '''
    Full Tree 
        
    Args:
        pset (PrimitivSet): set of functions, terminals and variables
        max_depth (int): initialization max depth
        max_size (int): the max size of the array that contains the tree
    Returns:
        random tree in an array using the full method
    '''
    functions_idx = np.array(list(pset.functions.keys()))
    terminals_idx = np.array(list(pset.terminals.keys()))
    variables_idx = np.array(list(pset.variables.keys()))
    all_terminals_idx = np.concatenate([terminals_idx, variables_idx])
    
    def full(depth):
        if depth == 0:
            # return a terminal/variable since it's maximum tree depth
            idx = all_terminals_idx[np.random.randint(all_terminals_idx.size)]
            full.tree[full.position] = idx
        else:
            # return a function
            idx = functions_idx[np.random.randint(functions_idx.size)]
            fn, arity, types = pset.functions[idx]
            full.tree[full.position] = idx
            depth -= 1
            for a in range(arity):
                full.position += 1
                full(depth)
    
    full.position = 0
    full.tree = np.zeros(max_size, dtype=np.int64)
    full(max_depth)

    return full.tree
