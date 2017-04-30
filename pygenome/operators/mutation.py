import numpy as np
import pygenome as pg


def apply_mutation(pop, rate, mt_op, **kargs):
    '''
    Apply Mutation

    Args:
        pop (Population): individuals to apply mutation
        rate (float): mutation rate
        mt_op (function): mutation operator for 1 genome

    Return:
        population of individuals after mutation being applied in-place
    '''
    for i in range(0, pop.size):
        if np.random.uniform() < rate:
            offspring = mt_op(pop.individuals[i], **kargs)
            pop.individuals[i] = offspring
            pop.individuals[i].run_eval = True
    
    return pop


def binary_flip_mutation(ind, gene_rate=None, **kargs):
    '''
    Flip Mutation

    Args:
        ind (Individual): individual with integer chromossome to be mutated
        rate (float): per gene mutation rate
        
    Returns:
        mutated individual
    '''
    chromossome = ind.genotype
    rate = 1. / chromossome.size if gene_rate is None else gene_rate

    for i in range(chromossome.size):
        if np.random.uniform() < rate:
            if chromossome[i] == 0:
                chromossome[i] = 1
            else:
                chromossome[i] = 0    
    
    ind.genotype = chromossome

    return ind


def flip_mutation(ind, gene_rate=None, low=0, high=1, **kargs):
    '''
    Flip Mutation

    Args:
        ind (Individual): individual with integer chromossome to be mutated
        rate (float): per gene mutation rate
        low (int): low value the chromossome can have
        hight (int): high value (inclusive) the chromossome can have

    Returns:
        mutated individual
    '''
    chromossome = ind.genotype
    rate = 1. / chromossome.size if gene_rate is None else gene_rate

    for i in range(0, chromossome.size):
        high_value = high + 1 # to be inclusive
        if np.random.uniform() < rate:
            chromossome[i] = np.random.randint(low, high=high_value)
    
    ind.genotype = chromossome

    return ind


def swap_mutation(ind, gene_rate=None, **kargs):
    '''
    Swap Mutation

    Args:
        ind (Individual): individual with integer chromossome to be mutated
        rate (float): per gene mutation rate

    Returns:
        mutated individual with chromossome where each gene can be 
        swapped with another one so that permutations can be preserved 
    '''
    chromossome = ind.genotype
    rate = 1. / chromossome.size if gene_rate is None else gene_rate

    for i in range(0, chromossome.size):
        if np.random.uniform() < rate:
            j = np.random.randint(chromossome.size)
            temp = chromossome[i]
            chromossome[i] = chromossome[j]
            chromossome[j] = temp

    ind.genotype = chromossome

    return ind


def uniform_mutation(ind, gene_rate=None, low=0.0, high=1.0, **kargs):
    '''
    Uniform Mutation

    Args:
        ind (Individual): individual with integer chromossome to be mutated
        rate (float): per gene mutation rate

    Returns:
        mutated individual
    '''
    chromossome = ind.genotype
    rate = 1. / chromossome.size if gene_rate is None else gene_rate

    for i in range(0, chromossome.size):
        if np.random.uniform() < rate:
            chromossome[i] = np.random.uniform(low=low, high=high)

    ind.genotype = chromossome

    return ind


# ES functions
def apply_global_mutation(pop, pool_size, mt_op, **kargs):
    '''
    Apply Global Mutation

    Args:
        pop (Population): individuals to apply mutation, randomly picked until pool_size
        pool_size (int): number of individuals that will be applied mutation
        mt_op (function): mutation operator
        **kargs: keyword arguments for the mutation oeprator

    Return:
        population of mutated individuals 
    '''
    new_pop = pg.make_empty_population(pool_size)
    
    for i in range(0, pool_size):
        index = np.random.randint(pop.size)
        new_pop.individuals[i] = pg.Individual()
        new_pop.individuals[i] = mt_op(pop.individuals[index], **kargs)
        new_pop.individuals[i].run_eval = True
    
    return new_pop


def sigma_check(sigma, epsilon=1e-08):
    '''
    Sigma Check

    Args:
        sigma (float): sigma value to be bounded
        epsilon (flota): minimum vale for sigma
    
    Returns:
        bounded sigma
    '''
    return epsilon if sigma < epsilon else sigma


def uncorrelated_one_step_mutation(ind, epsilon=1e-08):
    '''
    Uncorrelated One Step Mutation

    Args:
        ind (Individual): individual to be mutated. Last gene is the sigma value
        epsilon (float): minimum value sigma can have

    Returns:
        mutated individual
    '''
    chromossome = ind.genotype

    tau = 1.0 / np.sqrt(chromossome.size - 1)
    sigma = sigma_check(chromossome[-1] * np.exp(tau * np.random.normal()), epsilon)

    offspring = np.empty(chromossome.size, dtype=chromossome.dtype)

    for i in range(chromossome.size):
        offspring[i] = chromossome[i] + sigma * np.random.normal()

    offspring[-1] = sigma # last is the self-adaptive rate, not part of the solution

    ind.genotype = offspring

    return ind


def uncorrelated_n_steps_mutation(ind, epsilon=1e-08):
    '''
    Uncorrelated N Steps Mutation

    Args:
        ind (Individual): individual to be mutated.  
        First half is the problem, second half sigma values
        epsilon (float): minimum value sigma can have

    Returns:
        mutated individual
    '''
    chromossome = ind.genotype

    n = int(chromossome.size / 2)
    tau1 = (1.0 / np.sqrt(2.0 * n)) * np.random.normal()
    tau2 = 1.0 / np.sqrt(2.0 * np.sqrt(n))
    
    sigmas = np.array(chromossome[n:])
    for i in range(sigmas.size):
        sigmas[i] = sigma_check(sigmas[i] * np.exp(tau1 + tau2 * np.random.normal()), epsilon)

    values = np.array(chromossome[:n])
    for i in range(values.size):
        values[i] = values[i] + sigmas[i] * np.random.normal()

    ind.genotype = np.concatenate((values, sigmas))

    return ind
    

def tree_point_mutation(i1, pset=None, gene_rate=None, **kargs):
    '''
    Tree Point Mutation

    Args:
        i1 (TreeIndividual): the individual to be mutated
        pset (PrimitiveSet): the set primitives allowed to be used

    Returns:
        mutated individual
    '''
    
    if pset.ephemeral_cache:
        terminals_keys = []
        for k in list(pset.terminals.keys()):
            if k in pset.ephemeral_cache:
                pass
            else:
                terminals_keys.append(k)
        terminals_idx = np.concatenate([np.array(terminals_keys), np.array(list(pset.ephemeral_constants.keys()))])
    else:
        terminals_idx = np.array(list(pset.terminals.keys()))
    variables_idx = np.array(list(pset.variables.keys()))
    all_terminals_idx = np.concatenate([terminals_idx, variables_idx])
                
    new_genotype = np.copy(i1.genotype)

    i = 0
    while i < new_genotype.size and new_genotype[i] != 0:
        
        if np.random.uniform() < gene_rate:
            primitive = new_genotype[i]
            # replace terminal/variable with another one 
            if primitive in pset.terminals or primitive in pset.variables or pset.ephemeral_constants:
                if pset.typed:
                    _, term_types = pset.terminals[primitive] if primitive in pset.terminals else (None, None)
                    _, ephm_types = pset.ephemeral_constants[primitive] if primitive in pset.ephemeral_constants else (None, None)
                    _, vars_types = pset.variables[primitive] if primitive in pset.variables else (None, None)
                    
                    if term_types is not None:
                        typed_terminals = pset.terminals_types[term_types[0]]
                    else:
                        typed_terminals = [] 

                    if ephm_types is not None:
                        typed_ephemerals = pset.terminals_types[ephm_types[0]]
                    else:
                        typed_ephemerals = []
              
                    if vars_types is not None:
                        typed_variables = pset.variables_types[vars_types[0]] 
                    else:
                        typed_variables = []

                    valid_terminals = np.concatenate([np.array(typed_terminals, dtype=int), np.array(typed_variables, dtype=int), np.array(typed_ephemerals, dtype=int)])
                    new_genotype[i] = valid_terminals[np.random.randint(valid_terminals.size)]
                else:
                    new_genotype[i] = all_terminals_idx[np.random.randint(all_terminals_idx.size)]                 
            # replace function with another one of the same arity
            elif primitive in pset.functions:
                # only functions of the same arity are valid to be used
                _, arity, types = pset.functions[primitive]
                    
                if pset.typed:
                    arity_functions = set(pset.arity_cache[arity])
                    typed_functions = set(pset.functions_types[types[0]]) # only cache by return type
                    # compute intersection of arity and typed functions
                    # and if result is not null, check if all arguments are equivalent
                    # otherwise, does not mutate
                    final_candidates = []

                    for fn_key in set.intersection(arity_functions, typed_functions):
                        _, _, candidate_types = pset.functions[fn_key]
                        if candidate_types == types:
                            final_candidates.append(fn_key)

                    if final_candidates != []:
                        valid_functions_idx = np.array(final_candidates)
                        new_genotype[i] = valid_functions_idx[np.random.randint(valid_functions_idx.size)]                 
                else:
                    valid_functions_idx = np.array(pset.arity_cache[arity])
                    new_genotype[i] = valid_functions_idx[np.random.randint(valid_functions_idx.size)]

        i +=1
    
    new_individual = pg.TreeIndividual(tree=new_genotype, depth=i1.depth, nodes=i1.nodes)
    return new_individual


def subtree_mutation(parent, pset=None, **kargs):
    '''
    SubTree Mutation

    Args:
        parent (TreeIndividual): the individual to be mutated
        pset (PrimitiveSet): the set primitives allowed to be used

    Returns:
        mutated individual
    '''
    def arraycopy(src, src_pos, dest, dest_pos, length):
        dest[dest_pos:dest_pos + length] = src[src_pos:src_pos + length]
    
    def get_primitive_type(primitive):
        if primitive in pset.ephemeral_constants:
            _, p_type = pset.ephemeral_constants[primitive]
        elif primitive in pset.terminals:
            _, p_type = pset.terminals[primitive]
        elif primitive in pset.variables:
            _, p_type = pset.variables[primitive]
        elif primitive in pset.functions:
            _, _, p_type = pset.functions[primitive]
        else:
            p_type = None
            raise AttributeError('This is a typed primitive set so types are required!')
        return p_type[0]
        
    offspring = pg.TreeIndividual(tree=np.zeros(parent.genotype.size, dtype=parent.genotype.dtype))
   
    # define tree cut points for subtree replacement
    start1 = np.random.randint(parent.nodes)
    end1 = pg.transverse_tree(pset, parent.genotype, start1)

    # if typed subtree must return the appropriate type
    if pset.typed:
        primitive = parent.genotype[start1]
        parent_type = get_primitive_type(primitive)
    else:
        parent_type = None
    
     # TODO: the default values to set the size of the generated  tree must be revised
    # and a proper mechanism to set these values on a per-problem case must be available
    # if typed set, start2 must be of the same type as start1
    subtree = pg.grow_tree(pset, parent.depth - 1, parent.depth, initial_type=parent_type)
    start2 = 0
    end2 = pg.transverse_tree(pset, subtree, start2)

    len1 = start1 + (end2 - start2) + (parent.nodes - end1)

    # produce offpsring 1
    arraycopy(parent.genotype, 0, offspring.genotype, 0, start1)
    num_elements = (end2 - start2)
    arraycopy(subtree, start2, offspring.genotype, start1, num_elements)
    num_elements = (parent.nodes - end1)
    arraycopy(parent.genotype, end1, offspring.genotype, start1 + (end2 - start2), num_elements)

    # update tree metrics
    offspring.depth, offspring.nodes = pg.count_tree_internals(pset, offspring.genotype) 
    
    if offspring.nodes <= parent.genotype.size:
        return offspring
    else:
        return parent
