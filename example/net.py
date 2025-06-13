import numpy as np

class net:
    def __init__(self, layer_size, weight_init_std = 0.01):
        w_list = []
        b_list = []
        if len(layer_size) < 2:
            print(f"Error: layer less 2")
            return
        
        for i in range(len(layer_size) - 1):
            w_list.append(np.random.rand(layer_size[i], layer_size[i + 1]) * weight_init_std)
            b_list.append(np.zeros(layer_size[i + 1]))

        print(f"net has {len(layer_size)} layers")

    def predict(self, x):
        
        pass

    def loss(self, x, t):
        pass

    def gradient(self, x, t):
        pass