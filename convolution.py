from layer import Layer
import numpy as np


class Convolution(Layer):
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3)/9

    def padding(self, image):
        height, width = image.shape

        for i in range(height-2):
            for j in range(width-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input 

        height, width = input.shape

        output = np.zeros((height-2, width-2, self.num_filters))

        for im_regions, i, j in self.padding(input):
            output[i, j] = np.sum(im_regions * self.filters, axis=(1, 2))
        return output

    def backprop(self, gradient_loss, learning_rate):
        gradient_loss_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.padding(self.last_input):
            for f in range(self.num_filters):
                gradient_loss_filters[f] += gradient_loss[i, j, f] * im_region

        self.filters -= learning_rate * gradient_loss_filters

        return None
