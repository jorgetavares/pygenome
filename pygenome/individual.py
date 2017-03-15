
class Individual(object):
    """
    Base class for all type of individuals
    """

    def __init__(self):
        self.fitness = None # this should be an object since fitness can be more than a simple value
        self.genome = None  # this should be an object that can be of any type 
