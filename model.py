import numpy as np
from nnfs.datasets import vertical_data
import nnfs.core
from matplotlib import pyplot as plt
from time import sleep


# Numpy isn't consistent with datatypes. This Solves that
nnfs.core.init()
np.random.seed(0)

'''
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]
'''


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.biases = 0.10 * np.random.randn(1, n_neurons)
        print(self.biases)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights.T) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftMax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)


class Loss:
    """
    CATEGORIAL CROSS-ENTROPY \n
    Loss function:
        -ln(thing)
    """
    def calculate(self, output, y):
        """
        :param y:  target
        """
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

    def accuracy(self, output, y):
        # Use the largest prediction from outputs of each neuron.
        predictions = np.argmax(np.array(output), axis=1)
        accuracy = np.mean(np.array(predictions == y))
        return accuracy*100


class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            # Funky numpy indexing
            # Syntax <type np.array>[row_indices_to_pick, column_indices_to_pick_from_each_row]
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        loss = -np.log(correct_confidences)
        return loss


X, y = vertical_data(100, 3)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()
# plt.close()

# Because of the given dataset, there are only 2 inputs i.e. x and y.
# NOTE: X and y are different variables.
#       X is a feature set i.e. (x,y) plot values
#       while y is the target set which is one-hot encoded
layer1 = LayerDense(2, 100)
activation1 = ActivationReLU()

layer2 = LayerDense(100, 3)
activation2 = ActivationSoftMax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

loss_function = LossCategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)
# print(activation2.output)
print(loss)
print(loss_function.accuracy(activation2.output, y))

lowest_loss = np.inf

best_layer1_weights = layer1.weights.copy()
best_layer1_biases = layer1.biases.copy()
best_layer2_weights = layer2.weights.copy()
best_layer2_biases = layer2.biases.copy()

# Here comes the training part
for i in range(50_000):
    layer1.weights += 0.10 * np.random.randn(100, 2)
    layer1.biases += 0.10 * np.random.randn(1, 100)
    layer2.weights += 0.10 * np.random.randn(3, 100)
    layer2.biases += 0.10 * np.random.randn(1, 3)

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = loss_function.calculate(activation2.output, y)
    accuracy = loss_function.accuracy(activation2.output, y)

    if loss < lowest_loss:
        print(f"Best Generation: {i}\tloss: {loss}\taccuracy: {accuracy}")
        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases = layer2.biases.copy()
        lowest_loss = loss

    else:
        layer1.weights = best_layer1_weights.copy()
        layer1.biases = best_layer1_biases.copy()
        layer2.weights = best_layer2_weights.copy()
        layer2.biases = best_layer2_biases.copy()


layer1.weights = best_layer1_weights.copy()
layer1.biases = best_layer1_biases.copy()
layer2.weights = best_layer2_weights.copy()
layer2.biases = best_layer2_biases.copy()

plt.scatter(X[:, 0], activation2.output[:, 0], c='#ff0000')
plt.scatter(X[:, 0], activation2.output[:, 1], c='#00ff00')
plt.scatter(X[:, 0], activation2.output[:, 2], c='#0000ff')
plt.show()
plt.close()
