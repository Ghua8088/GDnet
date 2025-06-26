from .base import LossFunction
from ..gpu import gpu

class KLDivergence(LossFunction):
    def __call__(self, p, q):  # p = true, q = predicted
        xp = gpu.xp
        eps = 1e-15
        p = xp.clip(p, eps, 1)
        q = xp.clip(q, eps, 1)
        return xp.sum(p * xp.log(p / q)) / p.shape[0]

    def derivative(self, p, q):
        xp = gpu.xp
        eps = 1e-15
        p = xp.clip(p, eps, 1)
        q = xp.clip(q, eps, 1)
        return -p / q / p.shape[0]
