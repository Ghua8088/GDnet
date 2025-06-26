from .base import ActivationFunction
from ..gpu import gpu
import warnings
def softmax(x, axis=-1):
    xp = gpu.xp
    e_x = xp.exp(x - xp.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
class Softmax(ActivationFunction):
    def apply(self, x):
        return softmax(x)
    def derivative(self, x):
        warnings.warn("Using simplified softmax derivative (1s). Ensure cross-entropy loss is used", RuntimeWarning)
        xp = gpu.xp
        return xp.ones_like(x)