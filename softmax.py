from layer import Layer
import numpy as np


class Softmax(Layer):
    def __init__(self, input_len, nodes):
        super().__init__()
        self.weights = np.random.randn(input_len, nodes)/input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return (exp/np.sum(exp, axis=0))

    def backprop(self, loss_gradient_out, learning_rate):
        for i, gradient in enumerate(loss_gradient_out):
            if gradient == 0:
                continue

            exponential_totals = np.exp(self.last_totals)

            sum_exponential_totals = np.sum(exponential_totals)

            gradient_out_totals = - exponential_totals[i] * exponential_totals / (sum_exponential_totals ** 2)
            gradient_out_totals[i] = exponential_totals[i] * (sum_exponential_totals - exponential_totals[i]) / (sum_exponential_totals ** 2)

            gradient_totals_weights = self.last_input
            gradient_totals_biases = 1
            gradient_totals_inputs = self.weights

            gradient_loss_totals = gradient * gradient_out_totals

            gradient_loss_weights = gradient_totals_weights[np.newaxis].T @ gradient_loss_totals[np.newaxis]
            gradient_loss_biases = gradient_loss_totals * gradient_totals_biases
            gradient_loss_inputs = gradient_totals_inputs @ gradient_loss_totals

            self.weights -= learning_rate * gradient_loss_weights
            self.biases -= learning_rate * gradient_loss_biases
            return gradient_loss_inputs.reshape(self.last_input_shape)
