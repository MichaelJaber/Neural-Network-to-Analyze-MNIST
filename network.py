import system

import numpy as np

# class Network(object):
#     def __init__(self,layers):
#         """ Initalizes the weights and biases with a random array
#          according to a normal distribuation, with a mean of 0,
#          and a variance of 1. The biases need only be taken from sizes[1:]
#          since there is no bias present in the first layer of the network. """
#         self.num_layers = len(layers)
#         self.sizes = layers
#         for y in layers[1:]:
#             self.biases = [np.random.randn(y,1)]          # Creates an array representing all of the biases of each layer, 
#                                                           # The input layer is not included when creating the bias array,
#                                                           # since the bias is used to help calculate the output of a layer. 

#         for x , y in zip(layers[:-1], layers[1:]):
#             self.weights = [np.random.randn(y , x)]          # Creates an array representing all of the weights between the nodes of each respective layer,
#                                                              # There are no weights protruding out of the output layer.


#     def forward_pass(self,a):
#         """ Performs a forward pass through the neural network with input,
#         a, applying the sigmoid function at each node """
#         for w , b in zip(self.weights, self.bias):
#             a = sigmoid(np.dot(w , a) + b)
#         return a

#     def SGD(batches, mini_batch_size, learning_rate, training_data, test_data = None):
#         """ Stochastic gradient descent takes small batches of the training
#         data, which will then be passed through the backpropagation algorithm.
#         New weights and biases will be generated, and the network will be updated
#         to reflect this. """
        
#         ## input cases for test_data

#         for i in range(batches):
#             random.shuffle(training_data)
#             for i in range(0, len(training_data), mini_batch_size):
#                 mini_batches = [training_data[i:i+mini_batch_size]]
#                 for mini_batch in mini_batches:
#                     self.update_network(mini_batch, learning rate)

#     def update_network(self, mini_batch, learning rate):
#         """ Updates the weights and biases of the network according to the new 
#         values calculated by the backpropogation algorithm. """
#         for w in self.weights:
#             grad_wrt_w = [np.zeros(w.shape)]
#         for b in self.biases:
#             grad_wrt_b = [np.zeros(b.shape)]
#         for 

#     def backpropogation(self, x, y):
#         for w in self.weights:
#             grad_wrt_w = [np.zeros(w.shape)]
#         for b in self.biases:
#             grad_wrt_b = [np.zeros(b.shape)]

#         activation = x
#         activations = [x]                                # activations refers to the weighted inputs of each respective level, after being passed through the sigmoid activation function
#         zs = []                                          # outputs refers the outputs of each respective level, excluding the sigmoid activation function
#         for w, b in zip(self.weights, self.biases):
#             z = np.dot(w, start_activation) + b          # z can be visualized as a weighted input into a layer in the network
#             z.append(zs)
#             activation = sigmoid(z)
#             zs.append(activation)

#         delta = cost_derivative(activation[-1], y) * deriv_sigmoid(zs[-1])

#         grad_wrt_b[-1] = delta
#         grad_wrt_w[-1] = np.dot(activation[-1], delta)

#         for i in range(2, self.num_layers):
#             delta = np.dot(self.weights[-l+1].transpose(), delta) * deriv_sigmoid(z[-l])
#             grad_wrt_b[-l] = delta
#             grad_wrt_w[-l] = np.dot(activations[-l-1], delta)
#         return(grad_wrt_b, grad_wrt_w)











#     def cost_derivative(activation, y):


# ###########################################################################################################################################

# def sigmoid(x):
#         """ Sigmoid function, used as activation for neurons """
#         return 1.0 / (1.0 + np.exp(-x))

def deriv_sigmoid(x)
        """ Derivative of sigmoid function, used in backpropagation """
        return (sigmoid(x))(1.0 - sigmoid(x))
        
