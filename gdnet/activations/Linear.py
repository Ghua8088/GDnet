from .base import ActivationFunction
from ..gpu import gpu
class Linear(ActivationFunction):
    def apply(self,x):
        return x
    def derivative(self,x):
        xp = gpu.xp
        return xp.ones_like(x)