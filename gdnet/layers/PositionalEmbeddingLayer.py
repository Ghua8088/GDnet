from ..gpu import gpu
from .base import Layer
class PositionalEmbeddingLayer(Layer):
    def __init__(self, max_len, embedding_dim):
        xp = gpu.xp
        self.embedding = xp.random.randn(max_len, embedding_dim).astype(xp.float32) * 0.01
        self.max_len = max_len
        self.embedding_dim = embedding_dim
    def forward(self, x):
        xp = gpu.xp
        B, T, D = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
        pos_embed = self.embedding[:T]  # (T, D)
        pos_embed = xp.broadcast_to(pos_embed, (B, T, D)).copy()
        return x + pos_embed
    def backward(self, grad_output, learning_rate, lambda_):
        B, T, D = grad_output.shape
        grad_pos = grad_output.sum(axis=0) / B
        self.embedding[:T] -= learning_rate * (grad_pos + lambda_ * self.embedding[:T])
        return grad_output