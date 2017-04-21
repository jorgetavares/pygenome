import numpy as np
import pygenome as pg


def apply_crossover(pop, rate, cx_op, **kargs):
    '''
    Apply Crossover

    Args:
        pop (Population): individuals to apply crossover, 2 by 2
        rate (float): crossover rate
        cx_op (function): crossover operator for 2 genomes
        **kargs: keyword arguments for the crossover oeprator

    Return:
        population of individuals after crossover being applied in-place
    '''
    for i in range(0, pop.size - 1, 2):
        if np.random.uniform() < rate:
            o1, o2 = cx_op(pop.individuals[i].genotype, pop.individuals[i+1].genotype, **kargs)
            pop.individuals[i].genotype = o1
            pop.individuals[i+1].genotype = o2
    
    return pop


def apply_tree_crossover(pop, rate, cx_op, **kargs):
    '''
    Apply Tree Crossover

    Args:
        pop (Population): individuals to apply crossover, 2 by 2
        rate (float): crossover rate
        cx_op (function): crossover operator for 2 tree individuals
        **kargs: keyword arguments for the crossover oeprator

    Return:
        population of individuals after crossover being applied in-place
    '''
    for i in range(0, pop.size - 1, 2):
        if np.random.uniform() < rate:
            o1, o2 = cx_op(pop.individuals[i], pop.individuals[i+1], **kargs)
            pop.individuals[i] = o1
            pop.individuals[i+1] = o2

    return pop


def one_point_crossover(g1, g2):
    '''
    One Point Crossover

    Args:
        g1 (array): indivdiual 1 fixed genome
        g2 (array): indivdiual 2 fixed genome
    
    Returns:
        tuple with resulting offsprings genomes
    '''
    cut_point = np.random.randint(g1.size)
    o1 = np.concatenate(g1[0:cut_point], g2[cut_point:])
    o2 = np.concatenate(g2[0:cut_point], g1[cut_point:])
    
    return (o1, o2)


def partially_match_crossover(g1, g2):
    '''
    One Point Order Crossover

    Args:
        g1 (array): indivdiual 1 fixed genome
        g2 (array): indivdiual 2 fixed genome
    
    Returns:
        tuple with resulting offsprings genomes
    '''
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

    return (o1, o2)


def uniform_crossover(g1, g2, rate=0.5):
    '''
    Uniform Crossover

    Args:
        g1 (array): indivdiual 1 fixed genome
        g2 (array): indivdiual 2 fixed genome
    
    Returns:
        tuple with resulting offsprings genomes
    '''

    o1 = np.empty(g1.size, dtype=g1.dtype)
    o2 = np.empty(g2.size, dtype=g2.dtype)
    
    for i in range(0, g1.size):
        if np.random.uniform() < rate:
            o1[i] = g1[i]
            o2[i] = g2[i]
        else:
            o1[i] = g2[i]
            o2[i] = g1[i]

    return (o1, o2)


# ES functions
def apply_global_crossover(pop, cx_op, **kargs):
    '''
    Apply Global Crossover

    Args:
        pop (Population): individuals to apply crossover, 1 by 1, where parent 2 is randomly picked
        cx_op (function): crossover operator for 2 genomes
        **kargs: keyword arguments for the crossover oeprator

    Return:
        population of individuals after crossover being applied in-place
    '''
    for i in range(0, pop.size):
        pop.individuals[i].genotype = cx_op(pop.individuals[i].genotype, 
                                          pop.individuals[np.random.randint(pop.size)].genotype, 
                                          **kargs)
    
    return pop


def intermediary_crossover(g1, g2):
    '''
    Intermediary Crossover

    Args:
        g1 (array): indivdiual 1 fixed genome
        g2 (array): indivdiual 2 fixed genome
    
    Returns:
        single offspring from 2 parents
    '''
    o1 = np.empty(g1.size, dtype=g1.dtype)

    for i in range(g1.size):
        o1[i] = (g1[i] + g2[i]) / 2.0

    return o1


def discrete_crossover(g1, g2):
    '''
    Discrete Crossover

    Args:
        g1 (array): indivdiual 1 fixed genome
        g2 (array): indivdiual 2 fixed genome
    
    Returns:
        single offspring from 2 parents
    '''
    o1 = np.empty(g1.size, dtype=g1.dtype)

    for i in range(g1.size):
        if np.random.uniform() < 0.5:
            o1[i] = g1[i]
        else:
            o1[i] = g2[i]

    return o1


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

    # create offsprings
    offspring1 = pg.TreeIndividual(tree=np.zeros(parent1.genotype.size, dtype=parent1.genotype.dtype))
    offspring2 = pg.TreeIndividual(tree=np.zeros(parent2.genotype.size, dtype=parent2.genotype.dtype))

    # define tree cut points for subtree swap
    start1 = np.random.randint(parent1.nodes)
    end1 = pg.transverse_tree(pset, parent1.genotype, start1)
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
