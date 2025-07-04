import numpy as np

def gradient_descent_1d(func, x):
    if len(x.shape) != 1:
        return
    grad = np.zeros_like(x)
    h = 1e-4

    for i in range(x.shape[0]):
        sx = x[i]
        x[i] = sx + h
        f_x1 = func(x[i])
        x[i] = sx - h
        f_x2 = func(x[i])
        x[i] = sx
        grad[i] = (f_x1 - f_x2) / (2 * h)
    
    return grad
        

def gradient_descent_2d(func, x):
    if len(x.shape) != 2:
        return
    grad = np.zeros_like(x)
    h = 1e-4

    for i in range(x.shape[0]):
        grad[i] = gradient_descent_1d(func, x[i])
    
    return grad
        
def gradient_descent(func, x):
    grad = np.zeros_like(x)
    h = 1e-4

    iter = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not iter.finished:
        idx = iter.multi_index
        sx = x[idx]
        x[idx] = sx + h
        f_x1 = func(x)  # f(x_i + h)

        x[idx] = sx - h
        f_x2 = func(x)  # f(x_i - h)

        x[idx] = sx

        grad[idx] = (f_x1 - f_x2) / (2 * h)
        iter.iternext()
    
    return grad