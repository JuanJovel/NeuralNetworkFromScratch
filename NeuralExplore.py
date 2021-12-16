import numpy as np

# Inputs are better when they are in batches.
# The batch size is the number of inputs.
# Increasing the batch size will make easier to train the network.
# Generally, in the range of 32 to 128.

# 4x1 vector inputs
# inputs = [1, 2, 3, 2.5]

# inputs as batch
# The inputs is now a 3x4 matrix.
inputs = [[1, 2, 3, 2.5],
					[2.0, 5.0, -1.0, 2.0],
					[-1.5, 2.7, 3.3, -0.8]]


# 3x4 matrix
weights = [ [0.2, 0.8, -0.5, 1.0],
						[0.5, -0.91, 0.26, -0.5],
						[-0.26, -0.27, 0.17, 0.87] ]

# Biases per neuron
biases = [2, 3, 0.5]



# To use the inputs as batch:
# Note that we cannot use the dot product as we have 2 3x4 matrices.
# Thus we need to use the transpose of the weights

# Convert the weights to a numpy array.
weights = np.array(weights)

# Transpose the weights
weights = weights.T

# Use numpy to calculate the dot product of the inputs and weights plus each bias.
# We then add the bias to each element of the resulting matrix.
layer_outputs = np.dot(inputs, weights) + biases

print(layer_outputs)
