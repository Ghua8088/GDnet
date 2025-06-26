import numpy as np
from gpu import gpu

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    def update(self, layer, grads):
        xp = gpu.xp
        self.t += 1
        if layer not in self.m:
            self.m[layer] = {}
            self.v[layer] = {}
            for param_name in grads:
                self.m[layer][param_name] = xp.zeros_like(grads[param_name])
                self.v[layer][param_name] = xp.zeros_like(grads[param_name])
        for param_name in grads:
            g = grads[param_name]
            m = self.m[layer][param_name]
            v = self.v[layer][param_name]
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)
            param = getattr(layer, param_name)
            param -= update