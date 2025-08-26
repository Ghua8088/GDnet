from .base import Layer
from ..gpu import gpu
class DropoutLayer(Layer):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.mask = None

    def forward(self, x, training=True):
        xp = gpu.xp 
        if training and self.dropout > 0.0:
            self.mask = xp.random.rand(*x.shape) >= self.dropout
            return x * self.mask.astype(xp.float32) / (1.0 - self.dropout)
        else:
            return x
    def backward(self, grad_output, learning_rate=0.0, lambda_=0.0):
        if self.mask is not None:
            return grad_output * self.mask.astype(grad_output.dtype) / (1.0 - self.dropout)
        return grad_output

    def get_config(self):
        return {"dropout": self.dropout}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass