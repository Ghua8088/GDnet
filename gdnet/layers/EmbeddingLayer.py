from ..gpu import gpu
from .base import Layer
class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, embedding_dim):
        xp = gpu.xp
        self.embedding = xp.random.randn(vocab_size, embedding_dim).astype(xp.float32) * 0.01
        self.input_ids = None

    def forward(self, x):  # x: (batch_size,)
        xp = gpu.xp
        x = x.astype(xp.int32)  # ✅ force correct index type
        self.input_ids = x
        return self.embedding[x]

    def backward(self, grad_output, learning_rate, lambda_):
        xp = gpu.xp
        grad_embed = xp.zeros_like(self.embedding)
        xp.add.at(grad_embed, self.input_ids, grad_output)
        self.embedding -= learning_rate * grad_embed
        return None