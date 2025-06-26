from .base import LossFunction
from ..gpu import gpu

class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, y_true, y_pred):
        xp = gpu.xp
        diff = y_true - y_pred
        abs_diff = xp.abs(diff)
        mask = abs_diff <= self.delta
        return xp.mean(xp.where(mask, 0.5 * diff**2, self.delta * (abs_diff - 0.5 * self.delta)))

    def derivative(self, y_true, y_pred):
        xp = gpu.xp
        diff = y_pred - y_true
        abs_diff = xp.abs(diff)
        return xp.where(abs_diff <= self.delta, diff, self.delta * xp.sign(diff)) / y_true.size