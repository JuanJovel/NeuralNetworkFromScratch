import numpy as np

np.random.seed(0)

class Layer_Dense:
    def __init__(self, numberOfInputs: int, numberOfNeurons: int):
        # Creates an appropriate random nInputsxnNeurons matrix.
        self.weights = 0.10 * np.random.randn(numberOfInputs, numberOfNeurons)
        self.biases = np.zeros((1, numberOfNeurons))

    def forward(self, inputs):
        # Calculates the dot product of the inputs and weights plus each bias.
        # We then add the bias to each element of the resulting matrix.
        self.output = np.dot(inputs, self.weights) + self.biases


X = [   [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8] ]

layer1 = Layer_Dense(4, 5)

layer2 = Layer_Dense(5, 2)

layer1.forward(X)

print("Layer 1:")
print(layer1.output)
print()

print("Layer 2:")
layer2.forward(layer1.output)
print(layer2.output)
