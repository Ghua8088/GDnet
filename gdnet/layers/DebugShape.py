from .base import Layer


class DebugShape(Layer):
    def __init__(self):
        pass
    def forward(self, x):
        print("DEBUG: Shape before Dense:", x.shape)
        return x
    def backward(self, grad, *args):
        return grad