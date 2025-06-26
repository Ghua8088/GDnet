from .base import ActivationFunction
from ..gpu import gpu
class Dropout(ActivationFunction):
    def __init__(self, p=0.5):
        assert 0 <= p < 1, "Dropout probability must be in [0,1)"
        self.p = p
        self.mask = None
        self.training = True  # Use this flag to distinguish train/test mode

    def apply(self, x):
        xp = gpu.xp
        if self.training:
            self.mask = xp.random.rand(*x.shape) > self.p
            return (x * self.mask) / (1 - self.p)
        else:
            return x