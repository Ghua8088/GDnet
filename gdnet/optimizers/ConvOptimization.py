import numpy as np
from ..gpu import gpu
from numpy.lib.stride_tricks import sliding_window_view

def im2col(X, filter_size, stride, padding):
    xp = gpu.xp
    batch_size, channels, height, width = X.shape
    f = filter_size
    if padding > 0:
        X_padded = xp.pad(X, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    else:
        X_padded = X
    X_padded_cpu = gpu.to_cpu(X_padded)
    windows = sliding_window_view(X_padded_cpu, (f, f), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    batch_size, channels, out_h, out_w, _, _ = windows.shape
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_h * out_w, -1)
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