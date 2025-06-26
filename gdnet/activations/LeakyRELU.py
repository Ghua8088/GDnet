from .base import ActivationFunction
from ..gpu import gpu
class LeakyRELU(ActivationFunction):
    def __init__(self, alpha=0.001):
        self.alpha = alpha
    def apply(self, x):
        xp = gpu.xp
        return xp.where(x < 0, self.alpha * x, x)
    def derivative(self, x):
        xp = gpu.xp
        return xp.where(x > 0, 1, self.alpha)
