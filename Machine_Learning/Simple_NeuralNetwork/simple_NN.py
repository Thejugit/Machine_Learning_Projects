import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def  sigmoid_derivative(x):
    return x * (1 - x)

training_input = np.array([[0, 0, 1],
                           [1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1]])
 
training_output = np.array([[0, 1, 1, 0]]).T

#np.random.seed(1)

synaptic_weights = np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for i in range(10000):
    input_layer = training_input
    dot_prod = np.dot(input_layer, synaptic_weights)
    output = sigmoid(dot_prod)
    error = training_output - output
    adjustments = error * sigmoid_derivative(output)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print("synaptic weights after training")
print(synaptic_weights)

print("output after training: ")
print(output)