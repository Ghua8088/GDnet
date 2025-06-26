import numpy as np
from .base import Layer
class Flatten(Layer):
    def __init__(self, input_shape=None):
        if input_shape is not None:
            self.output_size = np.prod(input_shape)
        self.input_shape = input_shape

    def forward(self, x):
        self.input_shape = x.shape
        self.output_size = np.prod(x.shape[1:])
        return x.reshape(x.shape[0], -1)
    def backward(self, grad_output, learning_rate, lambda_=0.0):
        return grad_output.reshape(self.input_shape)      