from .base import Layer
from ..gpu import gpu
from ..optimizers.ConvOptimization import im2col, col2im
HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as torch_F
    HAS_TORCH = True
except:
    pass
class Conv2DLayer:
    def __init__(self, input_shape, num_filters, filter_size, activation, stride=1, padding=0):
        self.conv = Conv2D(num_filters, filter_size, input_shape, stride, padding, use_torch_conv=True)
        self.activation = activation
        self.output_shape = self.conv.output_shape
    def forward(self, x):
        self.conv_out = self.conv.forward(x)
        return self.activation.apply(self.conv_out)
    def backward(self, grad_output, learning_rate, lambda_=0.0):
        grad_activation = grad_output * self.activation.derivative(self.conv_out)
        return self.conv.backward(grad_activation, learning_rate, lambda_)
class Conv2D(Layer):
    def __init__(self, num_filters, filter_size, input_shape, stride=1, padding=0, use_torch_conv=True):
        xp = gpu.xp
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.use_torch_conv = use_torch_conv and HAS_TORCH
        depth = input_shape[0]
        scale = xp.sqrt(2.0 / (filter_size * filter_size * depth))
        self.filters = xp.random.randn(num_filters, depth, filter_size, filter_size).astype(xp.float32) * scale
        self.biases = xp.zeros((num_filters,), dtype=xp.float32)
        in_h, in_w = input_shape[1], input_shape[2]
        out_h = (in_h - filter_size + 2 * padding) // stride + 1
        out_w = (in_w - filter_size + 2 * padding) // stride + 1
        self.output_shape = (num_filters, out_h, out_w)
    def forward(self, x):
        xp = gpu.xp
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_h, in_w = x.shape[2], x.shape[3]
        out_h = (in_h - self.filter_size + 2 * self.padding) // self.stride + 1
        out_w = (in_w - self.filter_size + 2 * self.padding) // self.stride + 1
        if self.use_torch_conv:
            x_torch = torch.from_numpy(gpu.to_cpu(x)).float()
            weight = torch.from_numpy(gpu.to_cpu(self.filters)).float()
            bias = torch.from_numpy(gpu.to_cpu(self.biases)).float()
            y = torch_F.conv2d(x_torch, weight, bias=bias, stride=self.stride, padding=self.padding)
            y_np = y.detach().cpu().numpy()
            y_np = y_np.astype(xp.float32)
            y_np =gpu.to_device(y_np)
            self.last_input = x
            self.last_input_shape = x.shape
            self.conv_out = y_np
            return y_np
        else:
            self.last_input = x
            self.last_input_shape = x.shape
            f = self.filter_size
            X_col, out_h, out_w = im2col(x, f, self.stride, self.padding)
            filters_col = self.filters.reshape(self.num_filters, -1)
            out_flat = X_col @ filters_col.T + self.biases.reshape(1, -1)
            out_flat = out_flat.reshape(x.shape[0], out_h, out_w, self.num_filters)
            out = out_flat.transpose(0, 3, 1, 2)
            self.conv_out = out
            return out
    def backward(self, d_out, learning_rate, lambda_=0.0):
        xp = gpu.xp
        if self.use_torch_conv:
            import torch
            import torch.nn.functional as torch_F
            x_torch = torch.from_numpy(gpu.to_cpu(self.last_input)).float().requires_grad_(True)
            weight = torch.from_numpy(gpu.to_cpu(self.filters)).float().requires_grad_(True)
            bias = torch.from_numpy(gpu.to_cpu(self.biases)).float().requires_grad_(True)
            d_out_torch = torch.from_numpy(gpu.to_cpu(d_out)).float()
            y = torch_F.conv2d(x_torch, weight, bias=bias, stride=self.stride, padding=self.padding)
            y.backward(d_out_torch)
            with torch.no_grad():
                weight -= learning_rate * weight.grad
                bias -= learning_rate * bias.grad
            self.filters = gpu.to_gpu(weight.detach().cpu().numpy()) if gpu._has_cuda else weight.detach().cpu().numpy()
            self.biases = gpu.to_gpu(bias.detach().cpu().numpy()) if gpu._has_cuda else bias.detach().cpu().numpy()
            grad_input = x_torch.grad.detach().cpu().numpy()
            grad_input = gpu.to_device(grad_input)
            return grad_input
        else:
            N, F, out_h, out_w = d_out.shape
            f = self.filter_size
            d_out_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, F)
            X_col, _, _ = im2col(self.last_input, f, self.stride, self.padding)
            filters_col = self.filters.reshape(F, -1)
            # Ensure all arrays are on the same device
            d_out_reshaped = gpu.to_device(d_out_reshaped)
            X_col = gpu.to_device(X_col)
            filters_col = gpu.to_device(filters_col)
            d_filters = d_out_reshaped.T @ X_col
            d_filters = d_filters.reshape(self.filters.shape)
            d_biases = xp.sum(d_out_reshaped, axis=0, keepdims=True).reshape(self.biases.shape)
            if lambda_ > 0:
                d_filters += lambda_ * self.filters
            dX_col = d_out_reshaped @ filters_col
            d_input = col2im(dX_col, self.last_input_shape, f, self.stride, self.padding)
            self.filters -= learning_rate * d_filters
            self.biases -= learning_rate * d_biases
            return d_input