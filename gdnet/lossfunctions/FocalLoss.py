from .base import LossFunction
from ..gpu import gpu

class FocalLoss(LossFunction):
    def __init__(self, gamma=2.0, alpha=1.0):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        xp = gpu.xp
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)
        pt = xp.sum(y_true * y_pred, axis=1)
        loss = -self.alpha * (1 - pt)**self.gamma * xp.log(pt)
        return xp.mean(loss)

    def derivative(self, y_true, y_pred):
        xp = gpu.xp
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)
        pt = xp.sum(y_true * y_pred, axis=1, keepdims=True)
        grad = self.alpha * self.gamma * (1 - pt)**(self.gamma - 1) * xp.log(pt) * y_pred
        grad -= self.alpha * (1 - pt)**self.gamma * y_true / pt
        return grad / y_true.shape[0]