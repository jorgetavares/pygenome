import numpy as np
import pygenome as pg


def apply_crossover(pop, rate, cx_op, **kargs):
    '''
    Apply Crossover

    Args:
        pop (Population): individuals to apply crossover, 2 by 2
        rate (float): crossover rate
        cx_op (function): crossover operator for 2 individuals
        **kargs: keyword arguments for the crossover operator

    Return:
        population of individuals after crossover being applied in-place
    '''
    for i in range(0, pop.size - 1, 2):
        if np.random.uniform() < rate:
            o1, o2 = cx_op(pop.individuals[i], pop.individuals[i+1], **kargs)
            pop.individuals[i] = o1
            pop.individuals[i+1] = o2
            pop.individuals[i].run_eval = True
            pop.individuals[i+1].run_eval = True
    
    return pop


def one_point_crossover(i1, i2):
    '''
    One Point Crossover

    Args:
        i1 (Individual): indivdiual 1 fixed genome
        i2 (Individual): indivdiual 2 fixed genome
    
    Returns:
        tuple with resulting offsprings
    '''
    cut_point = np.random.randint(i1.genotype.size)
    o1 = np.concatenate([i1.genotype[0:cut_point], i2.genotype[cut_point:]])
    o2 = np.concatenate([i2.genotype[0:cut_point], i1.genotype[cut_point:]])
    
    i1.genotype = o1
    i2.genotype = o2

    return (i1, i2)


def partially_match_crossover(i1, i2):
    '''
    One Point Order Crossover

    Args:
        i1 (Individual): indivdiual 1 fixed genome
        i2 (Individual): indivdiual 2 fixed genome
    
    Returns:
        tuple with resulting offsprings
    '''
    g1 = i1.genotype
    g2 = i2.genotype

    # compute start and end cut points
    point1 = np.random.randint(g1.size)
    point2 = np.random.randint(g1.size)
    start = min(point1, point2)
    end = max(point1, point2)
   
    # init offpsprings chromossomes
    o1 = np.full(g1.size, -1, dtype=g1.dtype)
    o2 = np.full(g2.size, -1, dtype=g2.dtype)
    o1_genes = set()
    o2_genes = set()

    # copy into offsprings the parents genes 
    for i in range(start, end):
        o1[i] = g2[i]
        o1_genes.add(g2[i])
        o2[i] = g1[i]
        o2_genes.add(g1[i])
    
    # compute genes that still need to pass
    genes1 = []
    genes2 = []
    for i in range(0, g1.size):
        if g1[i] not in o1_genes:
            genes1.append(g1[i])
        if g2[i] not in o2_genes:
            genes2.append(g2[i]) 
    
    # insert remaining genes
    for i in range(0, g1.size):
        if o1[i] == -1 and genes1:
            o1[i] = genes1.pop(0)
        if o2[i] == -1 and genes2:
            o2[i] = genes2.pop(0)

    i1.genotype = o1
    i2.genotype = o2
    
    return (i1, i2)


def uniform_crossover(i1, i2, rate=0.5):
    '''
    Uniform Crossover

    Args:
        i1 (Individual): indivdiual 1 fixed genome
        i2 (Individual): indivdiual 2 fixed genome
    
    Returns:
        tuple with resulting offsprings
    '''
    g1 = i1.genotype
    g2 = i2.genotype

    o1 = np.empty(g1.size, dtype=g1.dtype)
    o2 = np.empty(g2.size, dtype=g2.dtype)
    
    for i in range(0, g1.size):
        if np.random.uniform() < rate:
            o1[i] = g1[i]
            o2[i] = g2[i]
        else:
            o1[i] = g2[i]
            o2[i] = g1[i]
    
    i1.genotype = o1
    i2.genotype = o2

    return (i1, i2)


# ES functions
def apply_global_crossover(pop, cx_op, **kargs):
    '''
    Apply Global Crossover

    Args:
        pop (Population): individuals to apply crossover, 1 by 1, where parent 2 is randomly picked
        cx_op (function): crossover operator for 2 individuals
        **kargs: keyword arguments for the crossover operator

    Return:
        population of individuals after crossover being applied in-place
    '''
    for i in range(0, pop.size):
        pop.individuals[i] = cx_op(pop.individuals[i], 
                                   pop.individuals[np.random.randint(pop.size)], 
                                   **kargs)
        pop.individuals[i].run_eval = True
    
    return pop


def intermediary_crossover(i1, i2):
    '''
    Intermediary Crossover

    Args:
        i1 (Individual): indivdiual 1 fixed genome
        i2 (Individual): indivdiual 2 fixed genome
    
    Returns:
        single offspring from 2 parents
    '''
    g1 = i1.genotype
    g2 = i2.genotype

    o1 = np.empty(g1.size, dtype=g1.dtype)

    for i in range(g1.size):
        o1[i] = (g1[i] + g2[i]) / 2.0

    i1.genotype = o1

    return i1


def discrete_crossover(g1, g2):
    '''
    Discrete Crossover

    Args:
        i1 (Individual): indivdiual 1 fixed genome
        i2 (Individual): indivdiual 2 fixed genome
    
    Returns:
        single offspring from 2 parents
    '''
    g1 = i1.genotype
    g2 = i2.genotype

    o1 = np.empty(g1.size, dtype=g1.dtype)

    for i in range(g1.size):
        if np.random.uniform() < 0.5:
            o1[i] = g1[i]
        else:
            o1[i] = g2[i]
    
    i1.genotype = o1

    return i1


def tree_crossover(parent1, parent2, pset=None):
    '''
    Tree Crossover

    Args:
        parent1 (TreeIndividual): first parent
        parent2 (TreeIndividual): second parent
        pset (PrimitiveSet): the set primitives allowed to be used

    Returns:
        offsprings from the two parents
    '''     
    def arraycopy(src, src_pos, dest, dest_pos, length):
        dest[dest_pos:dest_pos + length] = src[src_pos:src_pos + length]

    def get_primitive_type(primitive):
        if primitive in pset.terminals:
            _, p_type = pset.terminals[primitive]
        elif primitive in pset.variables:
            _, p_type = pset.variables[primitive]
        elif primitive in pset.functions:
            _, _, p_type = pset.functions[primitive]
        else:
            p_type = None
            raise AttributeError('This is a typed primitive set so types are required!')
        return p_type[0]

    # create offsprings
    offspring1 = pg.TreeIndividual(tree=np.zeros(parent1.genotype.size, dtype=parent1.genotype.dtype))
    offspring2 = pg.TreeIndividual(tree=np.zeros(parent2.genotype.size, dtype=parent2.genotype.dtype))

    # define tree cut points for subtree swap
    start1 = np.random.randint(parent1.nodes)
    end1 = pg.transverse_tree(pset, parent1.genotype, start1)
    
    # if typed set, start2 must be of the same type as start1
    if pset.typed:
        p1_primitive = parent1.genotype[start1]
        p1_type = get_primitive_type(p1_primitive)
        valid_gene_pos = []
        pos = 0
        while pos < parent2.nodes:
            if p1_type == get_primitive_type(parent2.genotype[pos]):
                valid_gene_pos.append(pos)
            pos += 1

        if valid_gene_pos == []:
            return parent1, parent2 # there is not valid point in the other, return original parents
        else:
            valid_pos = np.array(valid_gene_pos)
            start2 = valid_pos[np.random.randint(valid_pos.size)]
    else:
        start2 = np.random.randint(parent2.nodes)
    end2 = pg.transverse_tree(pset, parent2.genotype, start2)
    
    # define length of offspring trees
    len1 = start1 + (end2 - start2) + (parent1.nodes - end1)
    len2 = start2 + (end1 - start1) + (parent2.nodes - end2)
    
    # produce offpsring 1
    arraycopy(parent1.genotype, 0, offspring1.genotype, 0, start1)
    num_elements = (end2 - start2)
    arraycopy(parent2.genotype, start2, offspring1.genotype, start1, num_elements)
    num_elements = (parent1.nodes - end1)
    arraycopy(parent1.genotype, end1, offspring1.genotype, start1 + (end2 - start2), num_elements)
    
    # produce offspring 2
    arraycopy(parent2.genotype, 0, offspring2.genotype, 0, start2)
    num_elements = (end1 - start1)
    arraycopy(parent1.genotype, start1, offspring2.genotype, start2, num_elements)
    num_elements = (parent2.nodes - end2) 
    arraycopy(parent2.genotype, end2, offspring2.genotype, start2 + (end1 - start1), num_elements)
    
    # update tree metrics
    offspring1.depth, offspring1.nodes = pg.count_tree_internals(pset, offspring1.genotype)
    offspring2.depth, offspring2.nodes = pg.count_tree_internals(pset, offspring2.genotype)

    return offspring1, offspring2
