from .base import Layer
from . import MultiHeadAttentionLayer, TransformerFeedForward
from ..gpu import gpu
class TransformerBlock(Layer):
    def __init__(self, input_size, num_heads, hidden_size, regularization=None):
        self.input_size = input_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.regularization = regularization
        self.attn = MultiHeadAttentionLayer(input_size, input_size, num_heads, regularization=regularization)
        self.ffn = TransformerFeedForward(input_size, hidden_size, regularization=regularization)
    def forward(self, x):
        self.x = x
        self.a_out = self.attn.forward(x)
        self.res1 = x + self.a_out
        self.f_out = self.ffn.forward(self.res1)
        self.out = self.res1 + self.f_out
        return self.out
    def backward(self, grad_output, lr, lambda_=0.0):
        grad_ffn = self.ffn.backward(grad_output, lr, lambda_)
        grad_res1 = grad_output + grad_ffn
        grad_attn = self.attn.backward(grad_res1, lr, lambda_)
        grad_input = grad_attn + self.x
        return grad_input
    def get_config(self):
        return {
            "input_size": self.input_size,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "regularization": self.regularization
        }
    def get_weights(self):
        return gpu.to_cpu(self.attn.get_weights())
    def set_weights(self, weights):
        self.attn.set_weights(gpu.to_device(weights))