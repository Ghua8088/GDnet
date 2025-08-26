import cupy as cp
import numpy as np
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
