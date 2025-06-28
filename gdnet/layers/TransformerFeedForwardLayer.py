from ..activations import RELU, Linear
from ..gpu import gpu
from . import DenseLayer
from .base import Layer

class TransformerFeedForward(Layer):
    def __init__(self, input_size, hidden_size, activation=RELU(), regularization='l2'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.regularization = regularization
        self.fc1 = DenseLayer(input_size, hidden_size, activation=Linear(), regularization=regularization)
        self.fc2 = DenseLayer(hidden_size, input_size, activation=Linear(), regularization=regularization)
    def forward(self, x):  # x: (B, S, D)
        xp = gpu.xp
        B, S, D = x.shape
        self.input_shape = (B, S, D)
        x_flat = x.reshape(-1, D)  # (B*S, D)
        self.h = self.fc1.forward(x_flat)  # (B*S, H)
        self.out_flat = self.fc2.forward(self.h)  # (B*S, D)
        return self.out_flat.reshape(B, S, D)
    def backward(self, grad_output, lr, lambda_=0.0):  # grad_output: (B, S, D)
        xp = gpu.xp
        B, S, D = grad_output.shape
        grad_output_flat = grad_output.reshape(-1, D)  # (B*S, D)
        grad_h = self.fc2.backward(grad_output_flat, lr, lambda_)  # (B*S, H)
        dx_flat = self.fc1.backward(grad_h, lr, lambda_)  # (B*S, D)
        return dx_flat.reshape(B, S, D)
    def get_config(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "regularization": self.regularization
        }
    def get_weights(self):
        return {
            "fc1": gpu.to_cpu(self.fc1.get_weights()),
            "fc2": gpu.to_cpu(self.fc2.get_weights())
        }

    def set_weights(self, weights):
        self.fc1.set_weights(gpu.to_device(weights["fc1"]))
        self.fc2.set_weights(gpu.to_device(weights["fc2"]))
