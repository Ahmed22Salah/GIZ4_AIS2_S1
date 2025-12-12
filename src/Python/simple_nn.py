import numpy as np

np.random.seed(0)

# Input data
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# Dense layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Tanh activation
class Activation_Tanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)


# Sigmoid activation
class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))


# Create layers and activations

layer1 = Layer_Dense(4, 5)      # Input → 5 neurons
activation1 = Activation_Tanh() # Layer1 activation

layer2 = Layer_Dense(5, 2)      # Layer1 → 2 neurons
activation2 = Activation_Sigmoid() # Layer2 activation

# Forward pass

layer1.forward(X)              # First dense layer
activation1.forward(layer1.output)

layer2.forward(activation1.output)  # Second dense layer takes activation1 as input
activation2.forward(layer2.output)

# Print results

print("Layer 1 output:\n", layer1.output)
print("\nActivation 1 (Tanh) output:\n", activation1.output)

print("\nLayer 2 output:\n", layer2.output)
print("\nActivation 2 (Sigmoid) output:\n", activation2.output)