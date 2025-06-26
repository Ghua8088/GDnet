from ..gpu import gpu
from .base import LossFunction
class MSE(LossFunction):
    def __call__(self, y_true, y_pred):
        xp = gpu.xp
        return ((y_true - y_pred) ** 2).mean()
    def derivative(self, y_true, y_pred):
        xp = gpu.xp
        return 2 * (y_pred - y_true) / y_true.size 