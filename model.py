import numpy as np
from nnfs.datasets import spiral

# Numpy isn't consistent with datatypes. This Solves that
# nnfs.init()


np.random.seed(0)

'''
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]
'''

X, y = spiral.create_data(100, 3)


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights.T) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)
