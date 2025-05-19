import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(func, x):
    h = 1e-4
    delta_y = (func(x + h) - func(x - h)) / (2 * h)
    return delta_y

def numerical_multi(func, x):
    # y = x0 ** 2 + x1 ** 2
    # dy/dx0 = 2 * x0
    # dy/dx1 = 2 * x1
    h = 1e-4
    delta_y = np.zero_like(x)
    for i in range(x.size()):
        # h(x + h)
        pass

def function_1(x):
    y = 0.01 * x**2 + 0.1 * x
    return y

def function_2(x):
    y = x[0]**2 + x[1]**2
    return y

def func_gradient_line(func, x):
    # y = kx + b
    k = numerical_diff(func, x)
    y = func(x)
    b = y - k * x 
    print(f"y = {k} * x + {b}")
    return k, b

def func_line(k, b, x):
    return k * x + b

def draw_func(func, draw_name):
    x = np.arange(0.0, 20.0, 0.1)
    y = func(x)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig(draw_name)


def draw_func_diff(func, diff_x, draw_name):
    x = np.arange(0.0, 20.0, 0.1)
    y = func(x)
    plt.plot(x, y)

    # dy = kx + b
    k, b = func_gradient_line(function_1, diff_x)
    dy = func_line(k, b, x)

    plt.plot(x, dy)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig(draw_name)
    plt.close()


def test():
    # draw_func(function_1, "draw/quad_func.png")
    # draw_func_diff(function_1, 5, "draw/quad_diff5_func.png")
    # draw_func_diff(function_1, 10, "draw/quad_diff10_func.png")
    # func_gradient_line(function_1, 5)
    a = np.array([3.0, 4.0])
    print(f"a = {a}\na.shape={a.shape}")
    b = np.zeros_like(a)
    print(f"a = {b}\na.shape={b.shape}")
    
    pass

def printDiff(x):
    print(f"f'({x}) = {numerical_diff(function_1, x)}")

if __name__ == "__main__":
    test()
    # printDiff(5)
    # printDiff(10)
