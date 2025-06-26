from ..activations import Softmax
from ..gpu import gpu
from .base import Layer


class MultiHeadAttention(Layer):
    def __init__(self,input_size, output_size,num_heads,activation=Softmax):
        xp = gpu.xp
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.head_dim = output_size // num_heads
        self.activation = activation() if isinstance(activation, type) else activation
        self.W_q = xp.random.randn(input_size, output_size) * 0.01
        self.W_k = xp.random.randn(input_size, output_size) * 0.01
        self.W_v = xp.random.randn(input_size, output_size) * 0.01
        self.W_o = xp.random.randn(output_size, output_size) * 0.01
        self.b_o = xp.zeros((1, output_size))
        self.b_q = xp.zeros((1, output_size))
        self.b_k = xp.zeros((1, output_size))
        self.b_v = xp.zeros((1, output_size))
    def split_heads(self, x, B, T):
        xp = gpu.xp
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    def combine_heads(self, x):
        xp = gpu.xp
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)
    def forward(self, x):
        xp = gpu.xp
        B, T, _ = x.shape
        self.x = x
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v
        Q = self.split_heads(Q, B, T)
        K = self.split_heads(K, B, T)
        V = self.split_heads(V, B, T)
        scores = xp.matmul(Q, K.transpose(0, 1, 3, 2)) / xp.sqrt(self.head_dim)
        mask = xp.tril(xp.ones((T, T)), k=-1).astype(bool)[None, None, :, :]
        scores = xp.where(mask, -1e9, scores)
        attention = self.activation.apply(scores)
        attended = xp.matmul(self.attention, V) 
        combined = self.combine_heads(attended)
        self.output = combined @ self.W_o + self.b_o
        self.Q = Q
        self.K = K
        self.V = V
        self.attention = attention
        self.attended = attended
        return self.output
    def backward(self, grad_output, learning_rate=0.01, lambda_=0.0):
        xp = gpu.xp
        B, T, _ = grad_output.shape
        dCombined = grad_output @ self.W_o.T 
        dW_o = self.combine_heads(self.attended).transpose(0, 2, 1) @ grad_output 
        dW_o = xp.mean(dW_o, axis=0) + lambda_ * self.W_o
        db_o = xp.mean(grad_output, axis=(0, 1), keepdims=True)
        dCombined = dCombined.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        dAttention = xp.matmul(dCombined, self.V.transpose(0, 1, 3, 2))
        dV = xp.matmul(self.attention.transpose(0, 1, 3, 2), dCombined)
        dScores = self.activation.derivative(dAttention, self.attention) / xp.sqrt(self.head_dim)
        dQ = xp.matmul(dScores, self.K) 
        dK = xp.matmul(dScores.transpose(0, 1, 3, 2), self.Q)  
        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, self.output_size)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, self.output_size)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T, self.output_size)
        dx_q = dQ @ self.W_q.T
        dx_k = dK @ self.W_k.T
        dx_v = dV @ self.W_v.T
        dx = dx_q + dx_k + dx_v  
        dW_q = xp.matmul(self.x.transpose(0, 2, 1), dQ)
        dW_k = xp.matmul(self.x.transpose(0, 2, 1), dK)
        dW_v = xp.matmul(self.x.transpose(0, 2, 1), dV)
        dW_q = xp.mean(dW_q, axis=0) + lambda_ * self.W_q
        dW_k = xp.mean(dW_k, axis=0) + lambda_ * self.W_k
        dW_v = xp.mean(dW_v, axis=0) + lambda_ * self.W_v
        db_q = xp.mean(dQ, axis=(0, 1), keepdims=True)
        db_k = xp.mean(dK, axis=(0, 1), keepdims=True)
        db_v = xp.mean(dV, axis=(0, 1), keepdims=True)
        self.W_q -= learning_rate * dW_q
        self.W_k -= learning_rate * dW_k
        self.W_v -= learning_rate * dW_v
        self.W_o -= learning_rate * dW_o
        self.b_q -= learning_rate * db_q
        self.b_k -= learning_rate * db_k
        self.b_v -= learning_rate * db_v
        self.b_o -= learning_rate * db_o
        return dx
class MultiHeadAttentionLayer(Layer):
    def __init__(self,input_size, output_size,num_heads,activation=Softmax):
        self.MultiHeadAttnLayer = MultiHeadAttention(input_size, output_size, num_heads, activation)
        self.num_heads = num_heads
        self.head_dim = output_size // num_heads 
    def forward(self, x):
        return self.MultiHeadAttnLayer.forward(x)
    def backward(self, grad_output, learning_rate, lambda_=0.0):
        return self.MultiHeadAttnLayer.backward(grad_output, learning_rate, lambda_)
