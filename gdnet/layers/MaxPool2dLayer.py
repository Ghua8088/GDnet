from ..gpu import gpu
from .base import Layer
import numpy as np

HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass
class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2, use_torch_pool=True):
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_torch_pool = use_torch_pool and HAS_TORCH

    def forward(self, x):
        self.input_shape = x.shape
        if self.use_torch_pool:
            x_torch = torch.from_numpy(gpu.to_cpu(x)).float()
            x_torch.requires_grad_(True)
            y, indices = F.max_pool2d(
                x_torch,
                kernel_size=self.kernel_size,
                stride=self.stride,
                return_indices=True
            )
            self.last_input = x_torch
            self.indices = indices
            return gpu.to_device(y.detach().cpu().numpy().astype(x.dtype))
        else:
            return self._manual_forward(x)

    def backward(self, grad_output, learning_rate=None, lambda_=None):
        if self.use_torch_pool:
            grad_output_torch = torch.from_numpy(gpu.to_cpu(grad_output)).float()
            with torch.no_grad():
                grad_input = F.max_unpool2d(
                    grad_output_torch,
                    self.indices,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    output_size=self.input_shape[2:]
                )
            return gpu.to_device(grad_input.cpu().numpy())
        else:
            return self._manual_backward(grad_output)

    def _manual_forward(self, x):
        xp = gpu.xp
        N, C, H, W = x.shape
        k, s = self.kernel_size, self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        self.input_shape = x.shape
        self.out_h, self.out_w = out_h, out_w

        self.cols = xp.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s, :, :]
        self.cols = self.cols.reshape(N, C, out_h, out_w, -1)
        self.max_indices = xp.argmax(self.cols, axis=-1)
        return xp.max(self.cols, axis=-1)

    def _manual_backward(self, grad_output):
        xp = gpu.xp
        N, C, H, W = self.input_shape
        k, s = self.kernel_size, self.stride
        out_h, out_w = self.out_h, self.out_w
        grad_input = xp.zeros((N, C, H, W), dtype=grad_output.dtype)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s
                        w_start = j * s
                        idx = self.max_indices[n, c, i, j]
                        h_offset, w_offset = divmod(idx, k)
                        grad_input[n, c, h_start + h_offset, w_start + w_offset] += grad_output[n, c, i, j]
        return grad_input

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


class MaxPool2DLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = MaxPool2D(kernel_size, stride, use_torch_pool=True)

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
        return {}

    def set_weights(self, weights):
        pass
