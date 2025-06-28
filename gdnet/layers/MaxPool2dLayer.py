from ..gpu import gpu
from .base import Layer
import numpy as np
class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
    def forward(self, x):
        xp = gpu.xp
        self.input_shape = x.shape
        N, C, H, W = x.shape
        k, s = self.kernel_size, self.stride

        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        self.out_h, self.out_w = out_h, out_w
        self.cols = xp.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s, :, :]
        self.cols = self.cols.reshape(N, C, out_h, out_w, -1)
        self.max_indices = xp.argmax(self.cols, axis=-1)
        out = xp.max(self.cols, axis=-1)

        return out
    def backward(self, grad_output, learning_rate=None, lambda_=None):
        xp = gpu.xp
        N, C, H, W = self.input_shape
        k, s = self.kernel_size, self.stride
        out_h, out_w = self.out_h, self.out_w
        grad_input = xp.zeros((N, C, H, W), dtype=grad_output.dtype)
        cols_reshaped = self.cols.reshape(N * C * out_h * out_w, -1)
        grad_flat = xp.zeros_like(cols_reshaped)
        idx = self.max_indices.flatten()
        grad_vals = grad_output.flatten()
        grad_flat[xp.arange(grad_flat.shape[0]), idx] = grad_vals
        grad_col = grad_flat.reshape(N, C, out_h, out_w, k, k)
        for i in range(k):
            for j in range(k):
                h_idx = xp.arange(out_h) * s + i
                w_idx = xp.arange(out_w) * s + j
                if xp is np:
                    grad_input[:, :, h_idx[:, None], w_idx] += grad_col[:, :, :, :, i, j]
                else:
                    h_idx = h_idx.reshape(1, 1, -1, 1)
                    w_idx = w_idx.reshape(1, 1, 1, -1)
                    grad_input = grad_input.copy()
                    grad_input = grad_input.at[:, :, h_idx, w_idx].add(grad_col[:, :, :, :, i, j])
        return grad_input
    def get_weights(self):
        return self.filters
    def set_weights(self, weights):
        self.filters = weights

class MaxPool2DLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.pool = MaxPool2D(kernel_size, stride)

    def forward(self, x):
        return self.pool.forward(x)

    def backward(self, grad_output, learning_rate, lambda_=0.0):
        return self.pool.backward(grad_output)
    def get_config(self):
        return {
            "kernel_size": self.kernel_size,
            "stride": self.stride
        }
    def get_weights(self):
        return gpu.to_cpu(self.pool.get_weights())
    def set_weights(self, weights):
        self.pool.set_weights(gpu.to_device(weights))