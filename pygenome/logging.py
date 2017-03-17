import numpy as np
import pygenome.population as population
import pygenome.fitness as fitness


def evolution_progress(generation, pop):
    '''
    Evolution Progress

    Args:
        generation (int): iteration number
        pop (Population): population to compute current statistics

    Returns:
        nothing since it prints basic stats
    '''
    samples = np.fromiter((i.fitness.value for i in pop.individuals), pop.individuals.dtype, pop.size)
    print('%s\t%s\t%s\t%s' % (generation, np.min(samples), np.mean(samples), np.std(samples)))
