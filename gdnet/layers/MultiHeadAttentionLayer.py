from ..activations import Softmax
from ..gpu import gpu
from .base import Layer


class MultiHeadAttention(Layer):
    def __init__(self, input_size, output_size, num_heads, activation=Softmax, regularization=None):
        xp = gpu.xp
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.head_dim = output_size // num_heads
        self.activation = activation() if isinstance(activation, type) else activation
        self.regularization = regularization

        def init(shape): return xp.random.randn(*shape).astype(xp.float32) * 0.01
        self.W_q = init((input_size, output_size))
        self.W_k = init((input_size, output_size))
        self.W_v = init((input_size, output_size))
        self.W_o = init((output_size, output_size))

        self.b_q = xp.zeros((1, output_size), dtype=xp.float32)
        self.b_k = xp.zeros((1, output_size), dtype=xp.float32)
        self.b_v = xp.zeros((1, output_size), dtype=xp.float32)
        self.b_o = xp.zeros((1, output_size), dtype=xp.float32)

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

        self.attention = self.activation.apply(scores)
        attended = xp.matmul(self.attention, V)
        combined = self.combine_heads(attended)
        self.output = combined @ self.W_o + self.b_o

        self.Q, self.K, self.V = Q, K, V
        self.attended = attended
        return self.output

    def backward(self, grad_output, learning_rate=0.01, lambda_=0.0):
        xp = gpu.xp
        B, T, _ = grad_output.shape

        # === dW_o ===
        combined = self.combine_heads(self.attended)
        dW_o = xp.mean(combined.transpose(0, 2, 1) @ grad_output, axis=0)
        if lambda_ > 0:
            if self.regularization == 'l2':
                dW_o += lambda_ * self.W_o
            elif self.regularization == 'l1':
                dW_o += lambda_ * xp.sign(self.W_o)

        db_o = xp.mean(grad_output, axis=(0, 1), keepdims=True)

        dCombined = grad_output @ self.W_o.T
        dCombined = dCombined.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # === Backward attention ===
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

        dW_q = xp.mean(self.x.transpose(0, 2, 1) @ dQ, axis=0)
        dW_k = xp.mean(self.x.transpose(0, 2, 1) @ dK, axis=0)
        dW_v = xp.mean(self.x.transpose(0, 2, 1) @ dV, axis=0)

        if lambda_ > 0:
            if self.regularization == 'l2':
                dW_q += lambda_ * self.W_q
                dW_k += lambda_ * self.W_k
                dW_v += lambda_ * self.W_v
            elif self.regularization == 'l1':
                dW_q += lambda_ * xp.sign(self.W_q)
                dW_k += lambda_ * xp.sign(self.W_k)
                dW_v += lambda_ * xp.sign(self.W_v)

        db_q = xp.mean(dQ, axis=(0, 1), keepdims=True)
        db_k = xp.mean(dK, axis=(0, 1), keepdims=True)
        db_v = xp.mean(dV, axis=(0, 1), keepdims=True)

        # === Update weights ===
        self.W_q -= learning_rate * dW_q
        self.W_k -= learning_rate * dW_k
        self.W_v -= learning_rate * dW_v
        self.W_o -= learning_rate * dW_o
        self.b_q -= learning_rate * db_q
        self.b_k -= learning_rate * db_k
        self.b_v -= learning_rate * db_v
        self.b_o -= learning_rate * db_o
        return dx
    def get_weights(self):
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o, "b_q": self.b_q, "b_k": self.b_k, "b_v": self.b_v, "b_o": self.b_o}
    def set_weights(self, weights):
        self.W_q = weights["W_q"]
        self.W_k = weights["W_k"]
        self.W_v = weights["W_v"]
        self.W_o = weights["W_o"]
        self.b_q = weights["b_q"]
        self.b_k = weights["b_k"]
        self.b_v = weights["b_v"]        
        self.b_o = weights["b_o"]
class MultiHeadAttentionLayer(Layer):
    def __init__(self, input_size, output_size, num_heads, activation=Softmax, regularization=None):
        self.output_size = output_size
        self.MultiHeadAttnLayer = MultiHeadAttention(input_size, output_size, num_heads, activation, regularization)
        self.num_heads = num_heads
        self.head_dim = output_size // num_heads
    def forward(self, x):
        return self.MultiHeadAttnLayer.forward(x)
    def backward(self, grad_output, learning_rate, lambda_=0.0):
        return self.MultiHeadAttnLayer.backward(grad_output, learning_rate, lambda_)
    def get_config(self):
        return {
            "output_size": self.output_size,
            "num_heads": self.num_heads,
        }
    def get_weights(self):
        return gpu.to_cpu(self.MultiHeadAttnLayer.get_weights())
    def set_weights(self, weights):
        self.MultiHeadAttnLayer.set_weights(gpu.to_device(weights))
