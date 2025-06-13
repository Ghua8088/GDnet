import numpy as np
import cupy as cp
import pickle
from numba import njit,prange
import warnings
from sklearn.metrics import classification_report
import time
from numpy.lib.stride_tricks import sliding_window_view
from cupyx.scipy.ndimage import convolve as cp_convolve
import cupyx.scipy.signal
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
import sys
# Ensure NLTK stopwords are downloaded
"""
    IMPORTANT NOTES:
    1. This code is optimized for GPU execution using CuPy. Ensure you have a compatible GPU and CuPy installed.
    2. The framework supports both GPU (CuPy) and CPU (NumPy) backends. It will automatically fall back to CPU if a GPU is not available.
    * Access the array library via `gpu.xp` for seamless switching between CuPy and NumPy.
    3. Debugging is handled by the `DebugLogger` class, which can log messages to a file. You can enable or disable logging as needed.
    4. The `Model` class supports configurable architectures using layers like:
    - `DenseLayer`
    - `Conv2DLayer`
    - `MaxPool2DLayer`
    - `Flatten`
    5. Multiple activation and loss functions are provided as callable classes:
    - Activations: `RELU`, `Sigmoid`, `Softmax`, `LeakyRELU`
    - Losses: `MSE`, `CrossEntropy`
    6. For text-based tasks, the `TextManager` class provides preprocessing and TF-IDF vectorization (limited use cases only).
    7. Designed for research and educational use. While functional, it may lack the optimizations and features of production-grade deep learning libraries.
"""
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    import torch
    import torch.nn.functional as torch_F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
# EXPORTED API — Only these classes/functions are public.
# When adding new features, update __all__ if it's meant for external use.
#EXPOSED CLASSES ** IMPORTANT ** CHANGE THIS SAFELY
__all__ = [
    "Model", "DenseLayer", "Conv2DLayer", "MaxPool2DLayer", "Flatten",
    "TextManager", "RELU", "Softmax", "Sigmoid", "LeakyRELU",
    "Dropout", "CrossEntropy", "MSE", "LayerConfig", "DebugLogger"
]
# === DEBUG LOGGER ===
class DebugLogger:
    """ A simple logger that writes debug messages to a file.
        To use this logger , There already exists a DebugLogger instance  in model class in GDnet.py
        You can use it like this:
             self.logger.log("Your message here", "INFO")
    """
    def __init__(self, enabled=True, to_file=True, log_level="INFO"):
        self.enabled = enabled
        self.to_file = to_file
        self.log_level = log_level.upper()
        self.level_order = ["DEBUG", "INFO", "WARNING", "ERROR"]
        self.log_path = f"debug_{int(time.time())}.log"
        self.max_size = 100 * 1024 * 1024  # 100 MB
        self.log_file = None
        if self.to_file:
            try:
                self.log_file = open(self.log_path, "w", encoding="utf-8")
                self._write(f"[INFO] Logger initialized at {time.ctime()}")
            except Exception as e:
                self.to_file = False
                self.enabled = False
                raise RuntimeError(f"Failed to initialize log file: {e}")
    def log(self, message, level="INFO"):
        if not self.enabled or not self._should_log(level):
            return
        formatted = f"[{level}] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}"
        self._write(formatted)
    def _write(self, line):
        if self.to_file and self.log_file:
            try:
                if self.log_file.tell() < self.max_size:
                    self.log_file.write(line + "\n")
                    self.log_file.flush()
                else:
                    self._write("[ERROR] Log file exceeded max size. Disabling logging.")
                    self.to_file = False
                    self.enabled = False
            except Exception as e:
                self.to_file = False
                self.enabled = False
                raise RuntimeError(f"Logger write failed: {e}")

    def _should_log(self, level):
        try:
            return self.level_order.index(level.upper()) >= self.level_order.index(self.log_level)
        except ValueError:
            return False

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
            except:
                pass
# === Normalization Functions ===
def softmax(x):
    xp = gpu.xp
    e_x = xp.exp(x - xp.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
def sigmoid(x):
    xp = gpu.xp
    return 1 / (1 + xp.exp(-x))
def log_softmax(x):
    xp = gpu.xp
    c = xp.max(x, axis=1, keepdims=True)
    log_sum_exp = xp.log(xp.sum(xp.exp(x - c), axis=1, keepdims=True)) + c
    return x - log_sum_exp
def sparsemax(X):
    xp = gpu.xp
    output = xp.zeros_like(X)
    for i in range(X.shape[0]):
        z = X[i]
        z_sorted = xp.sort(z)[::-1]
        z_cumsum = xp.cumsum(z_sorted)
        k = xp.arange(1, z.size + 1)
        condition = 1 + k * z_sorted > z_cumsum
        k_z = k[condition][-1]
        tau = (z_cumsum[k_z - 1] - 1) / k_z
        output[i] = xp.maximum(z - tau, 0)
    return output
def temperature_scaled_softmax(x, T=1.0):
    xp = gpu.xp
    assert T > 0, "Temperature must be positive"
    x_temp = x / T
    e_x = xp.exp(x_temp - xp.max(x_temp, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# === Regularization functions===
def l2_regularization(weights, lambda_):
    xp = gpu.xp
    return lambda_ * sum(xp.sum(w**2) for w in weights)

def l1_regularization(weights, lambda_):
    xp = gpu.xp
    return lambda_ * sum(xp.sum(xp.abs(w)) for w in weights)
#=== Layer Optimization ===
def im2col(X, filter_size, stride, padding):
    xp = gpu.xp
    batch_size, channels, height, width = X.shape
    f = filter_size
    if padding > 0:
        X_padded = xp.pad(X, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    else:
        X_padded = X
    # sliding_window_view is only available in numpy
    X_padded_cpu = gpu.to_cpu(X_padded)
    windows = sliding_window_view(X_padded_cpu, (f, f), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    batch_size, channels, out_h, out_w, _, _ = windows.shape
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_h * out_w, -1)
    # Do NOT move cols to GPU here; keep on same device as input. Let the caller handle device.
    return cols, out_h, out_w

def col2im(cols, X_shape, filter_size, stride, padding):
    xp = gpu.xp
    N, C, H, W = X_shape
    f = filter_size
    out_h = (H + 2 * padding - f) // stride + 1
    out_w = (W + 2 * padding - f) // stride + 1
    H_padded, W_padded = H + 2*padding, W + 2*padding
    X_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    cols_reshaped = cols.reshape(N, out_h, out_w, C, f, f).transpose(0,3,4,5,1,2)
     
    X_padded_cpu = gpu.to_cpu(X_padded)
    cols_reshaped_cpu = gpu.to_cpu(cols_reshaped)
    for y in range(f):
        y_max = y + stride * out_h
        for x in range(f):
            x_max = x + stride * out_w
            np.add.at(X_padded_cpu,
                      (slice(None), slice(None), slice(y, y_max, stride), slice(x, x_max, stride)),
                      cols_reshaped_cpu[:, :, y, x, :, :])
    X_padded = gpu.to_device(X_padded_cpu)
    if padding == 0:
        return X_padded
    return X_padded[:, :, padding:-padding, padding:-padding]
 # ===Activation Functions ===
class ActivationFunction:
    def apply(self, x):
        raise NotImplementedError
    def derivative(self, x):
        raise NotImplementedError
class Dropout(ActivationFunction):
    def __init__(self, p=0.5):
        assert 0 <= p < 1, "Dropout probability must be in [0,1)"
        self.p = p
        self.mask = None
        self.training = True  # Use this flag to distinguish train/test mode

    def apply(self, x):
        xp = gpu.xp
        if self.training:
            # Create mask with zeros at dropped positions
            self.mask = xp.random.rand(*x.shape) > self.p
            # Scale output to keep expected value same during training
            return (x * self.mask) / (1 - self.p)
        else:
            # During evaluation, do nothing
            return x

    def derivative(self, x):
        xp = gpu.xp
        if self.training and self.mask is not None:
            # Gradient flows only through the kept neurons
            return self.mask / (1 - self.p)
        else:
            return xp.ones_like(x)
class RELU(ActivationFunction):
    def apply(self, x):
        xp = gpu.xp
        return xp.maximum(0, x)
    def derivative(self, x):
        xp = gpu.xp
        return xp.where(x > 0, 1, 0)

class LeakyRELU(ActivationFunction):
    def __init__(self, alpha=0.001):
        self.alpha = alpha
    def apply(self, x):
        xp = gpu.xp
        return xp.where(x < 0, self.alpha * x, x)
    def derivative(self, x):
        xp = gpu.xp
        return xp.where(x > 0, 1, self.alpha)

class Sigmoid(ActivationFunction):
    def apply(self, x):
        xp = gpu.xp
        return 1 / (1 + xp.exp(-x))
    def derivative(self, x):
        xp = gpu.xp
        s = self.apply(x)
        return s * (1 - s)
class Linear(ActivationFunction):
    def apply(self,x):
        return x
    def derivative(self,x):
        xp = gpu.xp
        return xp.ones_like(x)
class Softmax(ActivationFunction):
    def apply(self, x):
        return softmax(x)
    def derivative(self, x):
        warnings.warn("Using simplified softmax derivative (1s). Ensure cross-entropy loss is used", RuntimeWarning)
        xp = gpu.xp
        return xp.ones_like(x)
# === Loss Function ===
class LossFunction:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError
class MSE(LossFunction):
    def __call__(self, y_true, y_pred):
        xp = gpu.xp
        return ((y_true - y_pred) ** 2).mean()
    def derivative(self, y_true, y_pred):
        xp = gpu.xp
        return 2 * (y_pred - y_true) / y_true.size  
class CrossEntropy(LossFunction):
    def __init__(self, weight=None):
        self.weight = None
        if weight is not None :
            self.weight =gpu.to_device(weight)
    def __call__(self, y_true, y_pred):
        xp = gpu.xp
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)

        if y_true.ndim != 2:
            raise ValueError("y_true must be one-hot encoded")

        class_idx = xp.argmax(y_true, axis=1)
        if self.weight is not None:
            class_idx = xp.argmax(y_true, axis=1)
            weight_xp = cp.asarray(self.weight) if xp == cp else np.asarray(self.weight)
            sample_weights = weight_xp[class_idx]
            losses = -xp.sum(y_true * xp.log(y_pred), axis=1)
            return xp.mean(losses * sample_weights) / xp.mean(sample_weights)
        else:
            return -xp.sum(y_true * xp.log(y_pred)) / y_true.shape[0]
    def derivative(self, y_true, y_pred):
        xp = gpu.xp
        eps = 1e-15
        y_pred = xp.clip(y_pred, eps, 1 - eps)
        grad = y_pred - y_true
        if self.weight is not None:
            class_idx = xp.argmax(y_true, axis=1)
            weight_xp = cp.asarray(self.weight) if xp == cp else np.asarray(self.weight)
            sample_weights = weight_xp[class_idx]
            grad *= sample_weights[:, None] / xp.mean(sample_weights)
        return grad / y_true.shape[0]
# === Layer Architect Class ===
class LayerConfig:
    def __init__(self, layer_class, activation=None, **kwargs):
        self.layer_class = layer_class
        self.activation = activation
        self.kwargs = kwargs
# === Layer ===
class DenseLayer:
    def __init__(self, input_size, output_size, activation,regularization):
        xp = gpu.xp
        if isinstance(activation, (RELU, LeakyRELU)):
            scale = xp.sqrt(2.0 / input_size)
        else:
            scale = xp.sqrt(1.0 / input_size)
        self.w = xp.random.randn(input_size, output_size) * scale
        self.b = xp.zeros((1, output_size))
        self.activation = activation
        self.regularization=regularization

    def forward(self, x):
        xp = gpu.xp
        x= gpu.to_device(x)
        self.w=gpu.to_device(self.w)
        self.input = x
        self.z = xp.dot(x, self.w) + self.b
        self.a = self.activation.apply(self.z)
        return self.a
    def backward(self, grad_output, learning_rate,lambda_=0.0):
        xp = gpu.xp
        d_activation = self.activation.derivative(self.z)
        delta = grad_output * d_activation
        grad_w = xp.dot(self.input.T, delta)
        grad_b = xp.sum(delta, axis=0, keepdims=True)
        grad_input = xp.dot(delta, self.w.T)
        if lambda_ > 0:
            if self.regularization == 'l2':
                grad_w += lambda_ * self.w
            elif self.regularization == 'l1':
                grad_w += lambda_ * xp.sign(self.w)

        self.w -= learning_rate * grad_w
        self.b -= learning_rate * grad_b
        return grad_input
class MaxPool2D:
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
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                h_end = h_start + k
                w_start = j * s
                w_end = w_start + k
                grad_input[:, :, h_start:h_end, w_start:w_end] += grad_col[:, :, i, j, :, :]
        return grad_input


class MaxPool2DLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.pool = MaxPool2D(kernel_size, stride)

    def forward(self, x):
        return self.pool.forward(x)

    def backward(self, grad_output, learning_rate, lambda_=0.0):
        return self.pool.backward(grad_output)

class Conv2DLayer:
    def __init__(self, input_shape, num_filters, filter_size, activation, stride=1, padding=0):
        # Always force use_torch_conv=True
        self.conv = Conv2D(num_filters, filter_size, input_shape, stride, padding, use_torch_conv=True)
        self.activation = activation
        self.output_shape = self.conv.output_shape
    def forward(self, x):
        self.conv_out = self.conv.forward(x)
        return self.activation.apply(self.conv_out)
    def backward(self, grad_output, learning_rate, lambda_=0.0):
        grad_activation = grad_output * self.activation.derivative(self.conv_out)
        return self.conv.backward(grad_activation, learning_rate, lambda_)

def fast_conv2d_batch(x, filters, stride=1, padding=0):
     
     
    xp = gpu.xp
    batch, in_channels, H, W = x.shape
    out_channels, _, kH, kW = filters.shape
     
    if padding > 0:
        x = xp.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
    outs = []
    for i in range(out_channels):
         
        conv = None
        for c in range(in_channels):
             
            res = cupyx.scipy.signal.convolve(x[:, c], filters[i, c][None, :, :], mode='valid')
            if conv is None:
                conv = res
            else:
                conv += res
        if stride > 1:
            conv = conv[:, ::stride, ::stride]
        outs.append(conv)
    out = xp.stack(outs, axis=1)
    return out
class DebugShape:
    def forward(self, x):
        print("DEBUG: Shape before Dense:", x.shape)
        return x
    def backward(self, grad, *args):
        return grad
class Conv2D:
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
        self.biases = xp.zeros((num_filters,), dtype=xp.float32)  # 1D bias
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
            # Use PyTorch autograd for backward
            import torch
            import torch.nn.functional as torch_F
            # Convert everything to torch tensors
            x_torch = torch.from_numpy(gpu.to_cpu(self.last_input)).float().requires_grad_(True)
            weight = torch.from_numpy(gpu.to_cpu(self.filters)).float().requires_grad_(True)
            bias = torch.from_numpy(gpu.to_cpu(self.biases)).float().requires_grad_(True)
            d_out_torch = torch.from_numpy(gpu.to_cpu(d_out)).float()
            # Forward
            y = torch_F.conv2d(x_torch, weight, bias=bias, stride=self.stride, padding=self.padding)
            # Backward
            y.backward(d_out_torch)
            # Update weights
            with torch.no_grad():
                weight -= learning_rate * weight.grad
                bias -= learning_rate * bias.grad
            # Copy updated weights back
            self.filters = gpu.to_gpu(weight.detach().cpu().numpy()) if gpu._has_cuda else weight.detach().cpu().numpy()
            self.biases = gpu.to_gpu(bias.detach().cpu().numpy()) if gpu._has_cuda else bias.detach().cpu().numpy()
            # Return grad_input for next layer
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
class Flatten:
    def __init__(self, input_shape=None):
        if input_shape is not None:
            self.output_size = np.prod(input_shape)
        self.input_shape = input_shape

    def forward(self, x):
        self.input_shape = x.shape
        self.output_size = np.prod(x.shape[1:])
        return x.reshape(x.shape[0], -1)
    def backward(self, grad_output, learning_rate, lambda_=0.0):
        return grad_output.reshape(self.input_shape)        
 
#=== Model ===
class Model:
    def __init__(self, layer_configs, input_size=None, regularization=None):
        self.layers = []
        self.regularization = regularization
        self.logger= DebugLogger(enabled=True, to_file=True)
        in_features = input_size
        self.logger.log(f"Initializing model with input size: {in_features}, regularization: {regularization}, layers: {len(layer_configs)}", "INFO")
        for i, config in enumerate(layer_configs):
            layer_cls = config.layer_class
            activation = config.activation
            kwargs = config.kwargs
            
            if layer_cls == DenseLayer:
                if in_features is None:
                    raise ValueError("input_size must be provided for Dense layers")
                out_features = kwargs["output_size"]
                layer = DenseLayer(in_features, out_features, activation, regularization)
                in_features = out_features
            elif layer_cls == Conv2DLayer:
                kwargs["input_shape"] = in_features   
                kwargs["activation"] = activation
                layer = Conv2DLayer(**kwargs)
                in_features = layer.output_shape 
            elif layer_cls == Flatten:
                layer = Flatten(input_shape=in_features)
                in_features = layer.output_size
            elif layer_cls == MaxPool2DLayer:
                layer = MaxPool2DLayer(**kwargs)
                self.layers.append(layer)
                c, h, w = in_features
                k = kwargs.get("kernel_size", 2)
                s = kwargs.get("stride", 2)
                h = (h - k) // s + 1
                w = (w - k) // s + 1
                in_features = (c, h, w)
                continue 
            elif layer_cls == DebugShape:
                layer = DebugShape()
                in_features = in_features
            else:
                raise ValueError(f"Unsupported layer class: {layer_cls}")
            self.layers.append(layer)           
    def forward(self, x):
        out = gpu.to_gpu(x)
        for layer in self.layers:
            out = layer.forward(out)
        return out
    def backward(self, loss_grad, learning_rate,lambda_=0.0):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate,lambda_)
    def predict(self, x):
        return gpu.to_cpu(self.forward(x))
    def custom_warning(self,message, category, filename,lineno, file=None, line=None):
        self.logger.log(f"[WARNING] {message} ({category.__name__}) at {filename}:{lineno}", "WARNING")
    def validate(self, X_train, y_train, X_test, y_test,
             learning_rate, epochs, batch_size,
             verbose, loss_fn,
             lambda_, warmup_epochs,
             early_stopping, patience):
        xp = gpu.xp
        # Check input types and shapes
        self.logger.log("Validating input data types and shapes...", "INFO")
        assert isinstance(X_train, (np.ndarray, cp.ndarray)), "X_train must be a NumPy or CuPy array"
        assert isinstance(y_train, (np.ndarray, cp.ndarray)), "y_train must be a NumPy or CuPy array"
        assert isinstance(X_test, (np.ndarray, cp.ndarray)), "X_test must be a NumPy or CuPy array"
        assert isinstance(y_test, (np.ndarray, cp.ndarray)), "y_test must be a NumPy or CuPy array"
        self.logger.log("Validating input data shapes...", "INFO")
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of samples"
        assert X_test.shape[0] == y_test.shape[0], "X_test and y_test must have the same number of samples"
        # Check valid range for hyperparameters
        self.logger.log("Validating hyperparameters...", "INFO")
        assert isinstance(learning_rate, (float, int)) and learning_rate > 0, "learning_rate must be a positive float"
        assert isinstance(epochs, int) and epochs > 0, "epochs must be a positive integer"
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
        assert isinstance(verbose, bool), "verbose must be a boolean"
        assert callable(loss_fn) or hasattr(loss_fn, '__call__'), "loss_fn must be callable"
        assert isinstance(lambda_, (float, int)) and lambda_ >= 0, "lambda_ must be a non-negative float"
        assert isinstance(warmup_epochs, int) and warmup_epochs >= 0, "warmup_epochs must be a non-negative integer"
        assert isinstance(early_stopping, bool), "early_stopping must be a boolean"
        assert isinstance(patience, int) and patience >= 0, "patience must be a non-negative integer"
    def train(self, X_train, y_train, X_test, y_test,
          learning_rate=0.001, epochs=100, batch_size=32,
          verbose=False, loss_fn=MSE,
          lambda_=0.001, warmup_epochs=5,
          early_stopping=False, patience=10,debug=False):
        self.logger.set_enabled(debug)
        warnings.showwarning = self.custom_warning
        if not gpu._has_cuda:
            self.logger.log("CUDA not available. Using CPU for training.", "WARNING")
        else:
            self.logger.log("CUDA available. Using GPU for training.", "INFO")
        self.validate(X_train, y_train, X_test, y_test,
                  learning_rate, epochs, batch_size,
                  verbose, loss_fn,
                  lambda_, warmup_epochs,
                  early_stopping, patience)
        self.loss_fn = loss_fn
        n_samples = X_train.shape[0]
        best_loss = float('inf')
        epochs_no_improve = 0
        X_train = gpu.to_gpu(X_train)
        y_train = gpu.to_gpu(y_train)
        X_test = gpu.to_gpu(X_test)
        y_test = gpu.to_gpu(y_test)

        orig_batch_size = batch_size

        for epoch in range(epochs):
            start_time = time.time()
            if verbose:
                self.logger.log(f"Starting epoch {epoch+1}/{epochs} with batch size {batch_size}", "INFO")
            indices = gpu.xp.arange(n_samples)
            indices = gpu.to_cpu(indices)
            np.random.shuffle(indices)

            total_loss = 0
            num_batches = int(np.ceil(n_samples / batch_size))
            batch_size_ok = False

            while not batch_size_ok:
                try:
                    print(f"\nEpoch {epoch+1}/{epochs} - Using batch size: {batch_size}")
                    for batch_idx, start in enumerate(range(0, n_samples, batch_size)):
                        end = start + batch_size
                        batch_indices = indices[start:end]
                        X_batch = X_train[batch_indices]
                        y_batch = y_train[batch_indices]
                        output = self.forward(X_batch)
                        loss = self.loss_fn(y_batch, output)
                        total_loss += loss
                        if epoch >= warmup_epochs:
                            if self.regularization == 'l2':
                                loss += l2_regularization([layer.w for layer in self.layers if hasattr(layer, 'w')], lambda_)
                            elif self.regularization == 'l1':
                                loss += l1_regularization([layer.w for layer in self.layers if hasattr(layer, 'w')], lambda_)
                        grad_loss = loss_fn.derivative(y_batch, output)
                        self.backward(grad_loss, learning_rate, lambda_)
                        percent = (batch_idx + 1) / num_batches
                        bar = '=' * int(30 * percent) + '-' * (30 - int(30 * percent))
                        elapsed = time.time() - start_time
                        eta = (elapsed / (batch_idx + 1)) * (num_batches - (batch_idx + 1))
                        sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{num_batches} - Loss: {gpu.to_cpu(loss):.4f} - ETA: {eta:.1f}s")
                        sys.stdout.flush()
                    batch_size_ok = True

                except Exception as e:
                    import traceback
                    if 'out of memory' in str(e).lower() or 'cudaErrorMemoryAllocation' in str(e):
                        print(f"\nCUDA OOM detected at batch size {batch_size}. Reducing batch size...")
                        batch_size = max(1, batch_size // 2)
                        if batch_size == 1:
                            print("Batch size reduced to 1 but still OOM. Exiting training.")
                            raise e
                        continue
                    else:
                        traceback.print_exc()
                        raise e

            avg_loss = total_loss / num_batches
            epoch_time = time.time() - start_time
            # === Batched inference on test set ===
            y_pred_batches = []
            for i in range(0, X_test.shape[0], batch_size):
                Xb = X_test[i:i+batch_size]
                try:
                    out = self.forward(Xb)
                    y_pred_batches.append(out)
                    del Xb, out
                except Exception as e:
                    print(f"[ERROR] during test batch {i}: {e}")
                    continue
            y_pred = gpu.xp.concatenate(y_pred_batches, axis=0)
            y_true_cpu = gpu.to_cpu(y_test)
            y_pred_cpu = gpu.to_cpu(y_pred)
            if y_true_cpu.shape[1] > 1:
                true_labels = np.argmax(y_true_cpu, axis=1)
                pred_labels = np.argmax(y_pred_cpu, axis=1)
                accuracy = np.mean(true_labels == pred_labels)
            else:
                accuracy = np.mean(np.abs(y_true_cpu - y_pred_cpu) < 0.5)

            sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} [{'='*30}] - {num_batches}/{num_batches} - {epoch_time:.1f}s - Loss: {gpu.to_cpu(avg_loss):.4f} - Acc: {accuracy:.4f}\n")
            sys.stdout.flush()
            if early_stopping:
                if avg_loss < best_loss - 1e-4:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Stopping early at epoch {epoch+1} due to no improvement.")
                        self.logger.log(f"Early stopping at epoch {epoch+1} - No improvement for {patience} epochs", "INFO")
                        break
            self.logger.log(f"Epoch {epoch+1}/{epochs} completed - Loss: {gpu.to_cpu(avg_loss):.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.1f}s", "INFO")
        y_pred_batches = []
        for i in range(0, X_test.shape[0], batch_size):
            Xb = X_test[i:i+batch_size]
            try:
                out = self.forward(Xb)
                y_pred_batches.append(out)
                del Xb, out
                if gpu._has_cuda:
                    cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"[ERROR] during final test batch {i}: {e}")
                self.logger.log(f"[ERROR] during final test batch {i}: {e}", "ERROR")
                continue
        y_pred = gpu.xp.concatenate(y_pred_batches, axis=0)
        self.accuracy(y_test, y_pred)
    def accuracy(self,y_true, y_pred):
        y_true_cpu = gpu.to_cpu(y_true)
        y_pred_cpu = gpu.to_cpu(y_pred)
        print(classification_report(y_true_cpu.argmax(axis=1), y_pred_cpu.argmax(axis=1)))
        self.logger.log(classification_report(y_true_cpu.argmax(axis=1), y_pred_cpu.argmax(axis=1)), "INFO")
        print(f"Test set size: {y_true_cpu.shape[0]}")
        if y_true_cpu.shape[1] > 1: 
            true_labels = np.argmax(y_true_cpu, axis=1)
            pred_labels = np.argmax(y_pred_cpu, axis=1)
            accuracy = np.mean(true_labels == pred_labels)
            print(f"Test Accuracy: {accuracy:.4f}")
            self.logger.log(f"Test Accuracy: {accuracy:.4f}", "INFO")
            cm = np.zeros((y_true_cpu.shape[1], y_true_cpu.shape[1]), dtype=int)
            for t, p in zip(true_labels, pred_labels):
                cm[t, p] += 1
            print("Confusion Matrix:")
            self.logger.log("Confusion Matrix:", "INFO")
            self.logger.log(str(cm), "INFO")
            print(cm)
        else:
            mse = np.mean((y_true_cpu - y_pred_cpu)**2)
            mae = np.mean(np.abs(y_true_cpu - y_pred_cpu))
            print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")
            self.logger.log(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}", "INFO")
        if self.loss_fn:
            xp = gpu.xp
            y_true_gpu = xp.array(y_true_cpu)
            y_pred_gpu = xp.array(y_pred_cpu)
            loss = self.loss_fn(y_true_gpu, y_pred_gpu)
            print(f"Final Loss: {gpu.to_cpu(loss):.4f}")

    def save(self, path):
        logger_backup = getattr(self, 'logger', None)
        if logger_backup:
            self.logger = None
        for layer in self.layers:
            for attr in ['w', 'b', 'filters', 'biases']:
                if hasattr(layer, attr):
                    arr = getattr(layer, attr)
                    if 'cupy' in str(type(arr)):
                        try:
                            setattr(layer, attr, arr.get())
                        except Exception as e:
                            if logger_backup:
                                logger_backup.log(f"[WARNING] Could not move {attr} to CPU for layer {layer.__class__.__name__}: {e}", "WARNING")
                            setattr(layer, attr, None)
                        try:
                            gpu.clear_memory()
                        except Exception:
                            pass
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            if logger_backup:
                logger_backup.log(f"✅ Model saved to {path}", "INFO")
        except Exception as e:
            print(f"[ERROR] Could not save model: {e}")
            if logger_backup:
                logger_backup.log(f"[ERROR] Could not save model: {e}", "ERROR")
        finally:
            if logger_backup:
                self.logger = logger_backup
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        for layer in model.layers:
            for attr in ['w', 'b', 'filters', 'biases']:
                if hasattr(layer, attr):
                    arr = getattr(layer, attr)
                    if gpu._has_cuda and 'numpy' in str(type(arr)):
                        try:
                            setattr(layer, attr, gpu.to_gpu(arr))
                        except Exception as e:
                            print(f"[WARNING] Could not move {attr} to GPU: {e}")
        return model
#=== Text Manager ===
class TextManager:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = list(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize,
            preprocessor=self.preprocess,
            ngram_range=(1, 2),
            stop_words='english',  
            max_features=1000,
        )
    def fit(self, texts):
        self.vectorizer.fit(texts)
    def preprocess(self, text_string):
        space_pattern = r'\s+'
        giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = r'@[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text)
        return parsed_text
    def tokenize(self, word):
        tokens = re.split(r'[^a-zA-Z]+',word.lower())
        return [
            self.stemmer.stem(t) for t in tokens
            if t and t not in self.stopwords
        ]
    def transform(self, texts):
        return self.vectorizer.transform(texts)
class GPUManager:
    """Minimal GPU manager with automatic fallback to CPU"""
    def __init__(self):
        self._has_cuda = False
        try:
            self._array_module = cp
            _ = cp.array([1, 2, 3]) + 1
            self._has_cuda = True
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU Memory Info (free, total) in GB:")
            free, total = cp.cuda.Device().mem_info
            print(f"{free / 1024**3:.2f} GB free / {total / 1024**3:.2f} GB total")
            print("CUDA initialized successfully")
        except:
            self._array_module = np
            print("Falling back to CPU mode")
    @property
    def xp(self):
        return self._array_module
    def to_device(self,arr):
        if gpu._has_cuda:
            if isinstance(arr, np.ndarray):
                return cp.asarray(arr)
        return arr
    def to_gpu(self, array):
        if self._has_cuda:
            if isinstance(array, cp.ndarray):
                return array
            return cp.asarray(array, order='C')
        return array

    def to_cpu(self, array):
        return cp.asnumpy(array) if self._has_cuda else array

    def clear_memory(self):
        if self._has_cuda:
            cp.get_default_memory_pool().free_all_blocks()

gpu = GPUManager()

