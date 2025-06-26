from .base import LossFunction
from ..gpu import gpu
class CrossEntropy(LossFunction):
    def __init__(self, weight=None):
        self.weight = None
        if weight is not None :
            self.weight =gpu.to_device(weight)
    def __call__(self, y_true, y_pred):
        xp = gpu.xp
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)

        if y_true.ndim != 2:
            raise ValueError("y_true must be one-hot encoded")

        class_idx = xp.argmax(y_true, axis=1)
        if self.weight is not None:
            class_idx = xp.argmax(y_true, axis=1)
            weight_xp = gpu.to_device(self.weight) 
            sample_weights = weight_xp[class_idx]
            losses = -xp.sum(y_true * xp.log(y_pred), axis=1)
            return xp.mean(losses * sample_weights) / xp.mean(sample_weights)
        else:
            return -xp.sum(y_true * xp.log(y_pred)) / y_true.shape[0]
    def derivative(self, y_true, y_pred):
        xp = gpu.xp
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)
        grad = y_pred - y_true
        if self.weight is not None:
            class_idx = xp.argmax(y_true, axis=1)
            weight_xp =  gpu.to_device(self.weight)
            sample_weights = weight_xp[class_idx]
            grad *= sample_weights[:, None] / xp.mean(sample_weights)
        return grad / y_true.shape[0]