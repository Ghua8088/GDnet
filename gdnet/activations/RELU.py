from .base import ActivationFunction
from ..gpu import gpu
class RELU(ActivationFunction):
    def apply(self, x):
        xp = gpu.xp
        return xp.maximum(0, x)
    def derivative(self, x):
        xp = gpu.xp
        return xp.where(x > 0, 1, 0)