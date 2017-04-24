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
        self.functions_types = {}    # key=type, value=list of functions that return the type
        self.terminals_types = {}    
        self.variables_types = {}
        self.arity_cache = {}
        self.primitives = set() # to verify duplicates

    def _addTypesCache(self, primitive_key, types, types_cache):
        return_type = types[0]
        if return_type in types_cache:
            types_cache[return_type].append(primitive_key)
        else:
            types_cache[return_type] = [primitive_key]

    def _addArityCache(self, primitive_key, arity):
        if arity in self.arity_cache:
            self.arity_cache[arity].append(primitive_key)
        else:
            self.arity_cache[arity] = [primitive_key]

    
    def addFunction(self, fn, arity, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')
        
        if fn not in self.primitives:
            self.primitives.add(fn)

            primitive = (fn, arity, types)
            self.num_primitives += 1
            self.functions[self.num_primitives] = primitive
            self._addArityCache(self.num_primitives, arity)

            if self.typed:
                self._addTypesCache(self.num_primitives, types, self.functions_types)

    def addTerminal(self, term, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed primitive set so types are required!')
        
        if term not in self.primitives:
            self.primitives.add(term)
            
            primitive = (term, types)
            self.num_primitives += 1
            self.terminals[self.num_primitives] = primitive

            if self.typed:
                self._addTypesCache(self.num_primitives, types, self.terminals_types)

    def addVariable(self, var, types=None):
        if self.typed and types is None:
            raise AttributeError('This is a typed variable set so types are required!')   
        
        if var not in self.primitives:
            self.primitives.add(var)
            
            variable = (var, types)
            self.num_primitives += 1
            self.variables[self.num_primitives] = variable

            if self.typed:
                self._addTypesCache(self.num_primitives, types, self.variables_types)


def transverse_tree(pset, array_tree, counter):
    '''
    Transverse Tree

    Args:
        pset (PrimitiveSet): set of functions, terminals and variables
        array_tree (array): tree encoded as an array
        counter (int): the starting position to transverse the tree
    Returns:
        the length of the transversed tree (or subtree)
    '''
    element = array_tree[counter]
    counter += 1

    if element in pset.terminals:
        return counter
    elif element in pset.variables:
        return counter
    elif element in pset.functions:
        fn, arity, _ = pset.functions[element]
        for i in range(arity):
            counter = transverse_tree(pset, array_tree, counter)
        return counter
    else:
        print('primitive=', element)
        print('counter=', counter)
        print('array_tree=', array_tree)
        raise AttributeError('Primitive not found in Primitive Set!')

    
def count_tree_internals(pset, tree):
    '''
    Count Tree Internals

    Args:
        pset (PrimitiveSet): set of functions, terminals and variables
        tree (array): tree encoded as an array

    Returns:
        tuple with the tree depth and the number of nodes
    '''
    def run_tree(array_tree, depth):
        element = array_tree[run_tree.position]
        run_tree.position += 1
        run_tree.total_nodes += 1
        depth += 1

        if element in pset.terminals:
            return [depth]
        elif element in pset.variables:
            return [depth]
        elif element in pset.functions:
            fn, arity, _ = pset.functions[element]
            results = []
            for i in range(arity):
                results = results + run_tree(array_tree, depth)
            return results
        else:
            print('primitive=', element)
            print('position=', run_tree.position)
            print('array_tree=', array_tree)
            raise AttributeError('Primitive not found in Primitive Set!')   
    
    run_tree.total_nodes = 0
    run_tree.position = 0
    depth = max(run_tree(tree, 0))

    return depth, run_tree.total_nodes


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


def max_size_from_tree_max_depth(pset, tree_max_depth):
    '''
    Max size from Tree Max Depth

    Args:
        pset (PrimitiveSet): set of functions, terminals and variables
        tree_max_depth (int): maximum tree depth allowed
    Returns:
        max total number of nodes possible given a tree depth 
        and the largest arity in the function set
    '''
    # total number of nodes = (N^L) / (N-1)
    # where N is the number of nodes (arity) and L is the tree depth
    max_arity = max(pset.arity_cache.keys())
    return int(pow(max_arity, tree_max_depth) / (max_arity - 1))


def grow_tree(pset, max_depth, total_max_depth, initial_type=None):
    '''
    Grow Tree 
        
    Args:
        pset (PrimitivSet): set of functions, terminals and variables
        max_depth (int): initialization max depth
        total_max_depth (int): max tree depth allowed
        initial_type (type): if typed, the initial type
    Returns:
        random tree in an array using the grow method
    '''
    max_size = max_size_from_tree_max_depth(pset, total_max_depth)
    functions_idx = np.array(list(pset.functions.keys()))
    terminals_idx = np.array(list(pset.terminals.keys()))
    variables_idx = np.array(list(pset.variables.keys()))
    all_terminals_idx = np.concatenate([terminals_idx, variables_idx])
    all_primitives_idx = np.concatenate([functions_idx, all_terminals_idx])

    def grow(depth, arg_type=None):
        if depth == 1:
            # return a terminal/variable since it's maximum tree depth
            if arg_type is None:
                idx = all_terminals_idx[np.random.randint(all_terminals_idx.size)]
            else:
                valid_terminals = []
                if arg_type in pset.terminals_types:
                    valid_terminals = valid_terminals + pset.terminals_types[arg_type]
                if arg_type in pset.variables_types:
                    valid_terminals = valid_terminals + pset.variables_types[arg_type]  

                if len(valid_terminals) == 0:
                    AttributeError('Typed Primitive not found in Terminal/Variable Set!')    

                idx = valid_terminals[np.random.randint(len(valid_terminals))]
            
            grow.tree[grow.position] = idx
        else:
            # return a function or a terminal/variable
            if arg_type is None:
                idx = all_primitives_idx[np.random.randint(all_primitives_idx.size)]
            else:
                valid_set = []
                if arg_type in pset.functions_types:
                    valid_set = valid_set + pset.functions_types[arg_type]  
                if arg_type in pset.terminals_types:
                    valid_set = valid_set + pset.terminals_types[arg_type]
                if arg_type in pset.variables_types:
                    valid_set = valid_set + pset.variables_types[arg_type]  

                if len(valid_set) == 0:
                    AttributeError('Typed Primitive not found in Terminal/Variable Set!')  
                idx = valid_set[np.random.randint(len(valid_set))]
                
            if idx in pset.terminals or idx in pset.variables:
                grow.tree[grow.position] = idx
            else:
                fn, arity, types = pset.functions[idx]
                grow.tree[grow.position] = idx
                depth -= 1
                for a in range(arity):
                    grow.position += 1
                    if pset.typed:
                        grow(depth, types[a + 1])   # only cares about the arguments types
                    else:
                        grow(depth)
        
    grow.position = 0
    grow.tree = np.zeros(max_size, dtype=np.int64)
    grow(max_depth, arg_type=initial_type)

    return grow.tree


def full_tree(pset, max_depth, total_max_depth, initial_type=None):
    '''
    Full Tree 
        
    Args:
        pset (PrimitivSet): set of functions, terminals and variables
        max_depth (int): initialization max depth
        total_max_depth (int): max tree depth allowed
        initial_type (type): if typed, the initial type
    Returns:
        random tree in an array using the full method
    '''
    max_size = max_size_from_tree_max_depth(pset, total_max_depth)
    functions_idx = np.array(list(pset.functions.keys()))
    terminals_idx = np.array(list(pset.terminals.keys()))
    variables_idx = np.array(list(pset.variables.keys()))
    all_terminals_idx = np.concatenate([terminals_idx, variables_idx])

    def full(depth, arg_type=None):
        if depth == 1:
            # return a terminal/variable since it's maximum tree depth
            if arg_type is None:
                idx = all_terminals_idx[np.random.randint(all_terminals_idx.size)]
            else:
                valid_terminals = []
                if arg_type in pset.terminals_types:
                    valid_terminals = valid_terminals + pset.terminals_types[arg_type]
                if arg_type in pset.variables_types:
                     valid_terminals = valid_terminals + pset.variables_types[arg_type]  

                if len(valid_terminals) == 0:
                    AttributeError('Typed Primitive not found in Terminal/Variable Set!')    

                idx = valid_terminals[np.random.randint(len(valid_terminals))]
            
            full.tree[full.position] = idx

        else:
            # return a function
            if arg_type is None:
                idx = functions_idx[np.random.randint(functions_idx.size)]
            else:
                if arg_type not in pset.functions_types:
                    AttributeError('Typed Primitive not found in Terminal/Variable Set!')    
                    
                valid_functions = pset.functions_types[arg_type]
                idx = valid_functions[np.random.randint(len(valid_functions))]
            
            fn, arity, types = pset.functions[idx]
            full.tree[full.position] = idx
            depth -= 1
            for a in range(arity):
                full.position += 1
                if pset.typed:
                    full(depth, types[a + 1])   # only cares about the arguments types
                else:
                    full(depth)
    
    full.position = 0
    full.tree = np.zeros(max_size, dtype=np.int64)
    full(max_depth, arg_type=initial_type)

    return full.tree


def ramped_half_and_half_tree(pset, max_depth, total_max_depth, initial_type=None):
    '''
    Ramped Half and Half

    Args:
        pset (PrimitivSet): set of functions, terminals and variables
        max_depth (int): initialization max depth
        total_max_depth (int): max tree depth allowed
        initial_type (type): if typed, the initial type
    Returns:
        random generated tree
    '''
    if np.random.uniform() < 0.5:
        return full_tree(pset, max_depth, total_max_depth, initial_type=initial_type)
    else:
        return grow_tree(pset, max_depth, total_max_depth, initial_type=initial_type)
