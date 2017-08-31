import system

import numpy as np

class Network(object):
    def __init__(self,layers):
        """ Initalizes the weights and biases with a random array
         according to a normal distribuation, with a mean of 0,
         and a variance of 1. The biases need only be taken from sizes[1:]
         since there is no bias present in the first layer of the network. """
        self.num_layers = len(layers)
        self.sizes = layers
        for y in layers[1:]:
            self.biases = [np.random.randn(y,1)]

        for x , y in zip(layers[:-1], layers[1:]):
            self.weights = [np.random.randn(y , x)]


    def forward_pass(self,a):
        """ Performs a forward pass through the neural network with input,
        a, and returns it as an output. """
        for w , b in zip(self.weights, self.bias):
            a = sigmoid(np.dot(w , a) + b)
        return a

    def SGD(batches, mini_batch_size, learning_rate, training_data, test_data = None):
        """ Stochastic gradient descent takes small batches of the training
        data, which will then be passed through the backpropagation algorithm.
        New weights and biases will be generated, and the network will be updated
        to reflect this. """
        
        for i in range(batches):
            for i in range(0, len(training_data), mini_batch_size):
                mini_batches = [training_data[i:i+mini_batch_size]]
                for mini_batch in mini_batches
















    def sigmoid(x):
        """ Sigmoid function, used as activation for neurons """
        return 1.0 / (1.0 + np.exp(-x))

    def deriv_sigmoid(x)
        """ Derivative of sigmoid function, used in backpropagation """
        return (sigmoid(x))(1.0 - sigmoid(x))
        
