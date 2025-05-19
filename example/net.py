import numpy as np

from tools import sigmoid

class net:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"] = np.zeros(output_size)

        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        pass

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        return z2
        pass

    def loss(self, x, t):
        y = self.predict(x)
        
        pass

    def accuray(self, x, t):
        pass

    def gradient(self, x, t):
        pass