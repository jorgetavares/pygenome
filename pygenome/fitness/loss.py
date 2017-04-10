import numpy as np


def mean_squared_error(x, y):
    '''
    Mean Squared Error (MSR)

    Args:
        x (array): array of predictions
        y (array): array of labels

    Returns:
        the mean squared error
    '''
    return np.sum(np.sqrt(np.square(x - y)))
