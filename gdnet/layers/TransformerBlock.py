from .base import Layer
from . import MultiHeadAttentionLayer, TransformerFeedForward
class TransformerBlock(Layer):
    def __init__(self, input_size, num_heads, hidden_size):
        self.attn = MultiHeadAttentionLayer(input_size, input_size, num_heads)
        self.ffn = TransformerFeedForward(input_size, hidden_size)
    def forward(self, x):
        self.x = x
        self.x = x[:, 0, :] 
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