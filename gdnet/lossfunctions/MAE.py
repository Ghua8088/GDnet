from .base import LossFunction
from ..gpu import gpu

class MAE(LossFunction):
    def __call__(self, y_true, y_pred):
        xp = gpu.xp
        return xp.mean(xp.abs(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        xp = gpu.xp
        return xp.sign(y_pred - y_true) / y_true.size