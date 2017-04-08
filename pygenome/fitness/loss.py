import numpy as np


def mean_squared_error(x, y):
    return np.sum(np.sqrt(np.square(x) - np.square(y)))
