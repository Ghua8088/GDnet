from .base import Layer
from ..gpu import gpu
from ..activations import RELU, LeakyRELU
class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation,regularization):
        xp = gpu.xp
        if isinstance(activation, (RELU, LeakyRELU)):
            scale = xp.sqrt(2.0 / input_size)
        else:
            scale = xp.sqrt(1.0 / input_size)
        self.w = xp.random.randn(input_size, output_size) * scale
        self.b = xp.zeros((1, output_size))
        self.activation = activation
        self.regularization=regularization
    def forward(self, x):
        xp = gpu.xp
        x= gpu.to_device(x)
        self.w=gpu.to_device(self.w)
        self.input = x
        self.z = xp.dot(x, self.w) + self.b
        self.a = self.activation.apply(self.z)
        return self.a
    def backward(self, grad_output, learning_rate,lambda_=0.0):
        xp = gpu.xp
        d_activation = self.activation.derivative(self.z)
        delta = grad_output * d_activation
        grad_w = xp.dot(self.input.T, delta)
        grad_b = xp.sum(delta, axis=0, keepdims=True)
        grad_input = xp.dot(delta, self.w.T)
        if lambda_ > 0:
            if self.regularization == 'l2':
                grad_w += lambda_ * self.w
            elif self.regularization == 'l1':
                grad_w += lambda_ * xp.sign(self.w)

        self.w -= learning_rate * grad_w
        self.b -= learning_rate * grad_b
        return grad_input