from .base import ActivationFunction
from ..gpu import gpu
class Sigmoid(ActivationFunction):
    def apply(self, x):
        xp = gpu.xp
        return 1 / (1 + xp.exp(-x))
    def derivative(self, x):
        xp = gpu.xp
        s = self.apply(x)
        return s * (1 - s)