import numpy as np
from utils import sigmoid, softmax
from utils import mean_squared_error, cross_entropy_error
from utils import gradient_descent

class net:
    def __init__(self, layer_size, weight_init_std = 0.01):
        self.w_list = []
        self.b_list = []
        if len(layer_size) < 2:
            print(f"Error: layer less 2")
            return
        
        for i in range(len(layer_size) - 1):
            self.w_list.append(np.random.rand(layer_size[i], layer_size[i + 1]) * weight_init_std)
            self.b_list.append(np.zeros(layer_size[i + 1]))

        print(f"net has {len(layer_size)} layers")

    def predict(self, x):
        y = x
        for i in range(len(self.w_list) - 1):
            y = np.dot(y, self.w_list[i]) + self.b_list[i]
            y = sigmoid(y)
        
        y = np.dot(y, self.w_list[-1]) + self.b_list[-1]
        y = softmax(y)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        l = mean_squared_error(y, t)
        return l

    def gradient(self, x, t):
        # grad 和 x没有关系，因为x(输入)是不会变的
        # 但传进gradient_descent的参数有w & b, 虽然w & b不会传进loss，但是w & b会改变，然后predict需要w&b的值，因此影响了loss
        lossW = lambda W : self.loss(x, t) 
        grad = {}
        for i in range(len(self.w_list)):
            grad["w" + str(i)] = gradient_descent(lossW, self.w_list[i])

        for i in range(len(self.b_list)):
            grad["b" + str(i)] = gradient_descent(lossW, self.b_list[i])
        
        return grad