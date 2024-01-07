from layer import Layer
import numpy as np


class MaxPool(Layer):
    def __init__(self):
        super().__init__()

    def padding(self, image):
        height, width, _ = image.shape

        new_height = height // 2
        new_width = width // 2

        for i in range(new_height):
            for j in range(new_width):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        height, width, num_filters = input.shape
        output = np.zeros((height//2, width//2, num_filters))

        for im_region, i, j in self.padding(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, gradient_loss_out):
        gradient_loss = np.zeros(self.last_input.shape)

        for im_region, i, j in self.padding(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if (im_region[i2, j2, f2] == amax[f2]):
                            gradient_loss[i*2+i2, j*2+j2,f2] = gradient_loss_out[i, j, f2]
                            break
        return gradient_loss
