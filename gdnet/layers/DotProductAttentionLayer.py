from ..activations import Softmax
from ..gpu import gpu
from .base import Layer
class DotProductAttention(Layer):
    def __init__(self,input_size, output_size,activation=Softmax):
        xp = gpu.xp
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation() if isinstance(activation, type) else activation
        self.W_q = xp.random.randn(input_size, output_size) * 0.01
        self.W_k = xp.random.randn(input_size, output_size) * 0.01
        self.W_v = xp.random.randn(input_size, output_size) * 0.01
        self.b_q = xp.zeros((1, output_size))
        self.b_k = xp.zeros((1, output_size))
        self.b_v = xp.zeros((1, output_size))
    def forward(self, x):
        xp = gpu.xp
        self.x = x
        self.Q = x @ self.W_q + self.b_q
        self.K = x @ self.W_k + self.b_k
        self.V = x @ self.W_v + self.b_v
        self.scores = xp.matmul(self.Q, self.K.transpose(0, 2, 1)) /xp.sqrt(self.output_size)
        mask = xp.tril(xp.ones_like(self.scores), k=-1).astype(bool)
        self.scores = xp.where(mask, -1e9, self.scores)
        self.attention = self.activation.apply(self.scores)
        self.output = xp.matmul(self.attention, self.V)
        return self.output
    def backward(self, grad_output, learning_rate, lambda_=0.0):
        xp = gpu.xp
        B, T, d = grad_output.shape
        dAttention = xp.matmul(grad_output, self.V.transpose(0, 2, 1))
        dV = xp.matmul(self.attention.transpose(0, 2, 1), grad_output)
        dScores = self.activation.derivative(dAttention, self.attention) / xp.sqrt(self.output_size)
        dQ = xp.matmul(dScores, self.K)
        dK = xp.matmul(dScores.transpose(0, 2, 1), self.Q)
        dx_q = xp.matmul(dQ, self.W_q.T)
        dx_k = xp.matmul(dK, self.W_k.T)
        dx_v = xp.matmul(dV, self.W_v.T)
        dx = dx_q + dx_k + dx_v
        dW_q = xp.matmul(self.x.transpose(0, 2, 1), dQ).sum(axis=0) / B + lambda_ * self.W_q
        dW_k = xp.matmul(self.x.transpose(0, 2, 1), dK).sum(axis=0) / B + lambda_ * self.W_k
        dW_v = xp.matmul(self.x.transpose(0, 2, 1), dV).sum(axis=0) / B + lambda_ * self.W_v
        db_q = dQ.sum(axis=(0, 1), keepdims=True) / B
        db_k = dK.sum(axis=(0, 1), keepdims=True) / B
        db_v = dV.sum(axis=(0, 1), keepdims=True) / B
        self.W_q -= learning_rate * dW_q
        self.W_k -= learning_rate * dW_k
        self.W_v -= learning_rate * dW_v
        self.b_q -= learning_rate * db_q
        self.b_k -= learning_rate * db_k
        self.b_v -= learning_rate * db_v
        return dx
    def get_weights(self):
        return {"W_q": gpu.to_cpu(self.W_q), "W_k": gpu.to_cpu(self.W_k), "W_v": gpu.to_cpu(self.W_v), "b_q": gpu.to_cpu(self.b_q), "b_k": gpu.to_cpu(self.b_k), "b_v": gpu.to_cpu(self.b_v)}
    def set_weights(self, weights):
        self.W_q = gpu.to_device(weights["W_q"])
        self.W_k = gpu.to_device(weights["W_k"])
        self.W_v = gpu.to_device(weights["W_v"])
        self.b_q = gpu.to_device(weights["b_q"])
        self.b_k = gpu.to_device(weights["b_k"])        
        self.b_v = gpu.to_device(weights["b_v"])
class DotProductAttentionLayer(Layer):
    def __init__(self,input_size, output_size,activation=Softmax):
        self.DotProductAttnLayer = DotProductAttention(input_size, output_size, activation)
    def forward(self, x):
        return self.DotProductAttnLayer.forward(x)
    def backward(self, grad_output, learning_rate, lambda_=0.0):        
        return self.DotProductAttnLayer.backward(grad_output, learning_rate, lambda_)
    def get_config(self):
        return {
            "output_size": self.output_size,
        }
    def get_weights(self):
        return self.DotProductAttnLayer.get_weights()
    def set_weights(self, weights):
        self.DotProductAttnLayer.set_weights(weights)