import numpy as np
import pygenome.individual as individual
import pygenome.population as population


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
            offspring = mt_op(pop.individuals[i].genome, **kargs)
            pop.individuals[i].genome = offspring
    
    return pop
