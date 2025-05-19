import numpy as np

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)
    pass

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
    pass