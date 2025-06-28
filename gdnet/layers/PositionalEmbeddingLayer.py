from ..gpu import gpu
from .base import Layer
class PositionalEmbeddingLayer(Layer):
    def __init__(self, max_len, embedding_dim, regularization=None):
        xp = gpu.xp
        self.embedding = xp.random.randn(max_len, embedding_dim).astype(xp.float32) * 0.01
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.regularization = regularization
    def forward(self, x):
        xp = gpu.xp
        B, T, D = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
        self.seq_len = T 
        self.batch_size = B
        pos_embed = self.embedding[:T] 
        return x + xp.broadcast_to(pos_embed, (B, T, D)) 
    def backward(self, grad_output, learning_rate, lambda_=0.0):
        xp = gpu.xp
        grad_pos = grad_output.sum(axis=0) / self.batch_size  
        if lambda_ > 0:
            if self.regularization == 'l2':
                grad_pos += lambda_ * self.embedding[:self.seq_len]
            elif self.regularization == 'l1':
                grad_pos += lambda_ * xp.sign(self.embedding[:self.seq_len])
        self.embedding[:self.seq_len] -= learning_rate * grad_pos
        return grad_output  

    def get_config(self):
        return {
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim,
            "regularization": self.regularization
        }
    def get_weights(self):
        return gpu.to_cpu(self.embedding)
    def set_weights(self, weights):
        self.embedding = gpu.to_device(weights)