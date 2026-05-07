"""
Tensor — GPU tensor backed by MTL::Buffer via sk_tensor_alloc.

Usage:
    t = Tensor.gpu(512, 1024)          # allocates MTLBuffer on device
    t.from_numpy(arr)                   # CPU → GPU copy
    arr = t.numpy()                     # GPU → CPU copy
    t.free()                            # release GPU buffer
"""

import ctypes, os
import numpy as np

# ── load dylib once ────────────────────────────────────────────────

_dylib = None

def _lib():
    global _dylib
    if _dylib is None:
        path = os.environ.get("SK_DYLIB", "build/libsk.dylib")
        _dylib = ctypes.CDLL(path)

        _dylib.sk_tensor_alloc.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
        _dylib.sk_tensor_alloc.restype  = ctypes.c_void_p

        _dylib.sk_tensor_free.argtypes = [ctypes.c_void_p]

        _dylib.sk_tensor_to_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _dylib.sk_tensor_from_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        _dylib.sk_tensor_shape.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _dylib.sk_tensor_shape.restype  = ctypes.c_int

        _dylib.sk_tensor_numel.argtypes = [ctypes.c_void_p]
        _dylib.sk_tensor_numel.restype  = ctypes.c_uint32

        _dylib.sk_dispatch_elementwise.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        _dylib.sk_dispatch_elementwise.restype  = ctypes.c_int

    return _dylib


class DType:
    f16=0; f32=1; i32=2; i8=3; bf16=4
    _sizes = {0:2, 1:4, 2:4, 3:1, 4:2}
    _np_map = {0: np.float16, 1: np.float32, 2: np.int32, 3: np.int8, 4: np.float16}

    @classmethod
    def itemsize(cls, dt): return cls._sizes.get(dt, 2)
    @classmethod
    def to_np(cls, dt): return cls._np_map.get(dt, np.float16)
    @classmethod
    def from_np(cls, d):
        if d == np.float16: return cls.f16
        if d == np.float32: return cls.f32
        if d == np.int32:   return cls.i32
        if d == np.int8:    return cls.i8
        return cls.f16


class Tensor:
    """GPU tensor. Holds a handle to sk_tensor_alloc's heap object."""

    def __init__(self, *shape, dtype: int = DType.f16, handle=None):
        self._handle = handle
        self._dtype  = dtype
        self._shape  = shape if shape else ()
        if handle is None and shape:
            self._handle = _lib().sk_tensor_alloc(
                shape[0] if len(shape) > 0 else 1,
                shape[1] if len(shape) > 1 else 1,
                shape[2] if len(shape) > 2 else 1,
                shape[3] if len(shape) > 3 else 1,
                dtype)

    @classmethod
    def gpu(cls, *shape, dtype: int = DType.f16):
        """Allocate a GPU tensor."""
        return cls(*shape, dtype=dtype)

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        """Upload numpy array to GPU."""
        t = cls(*arr.shape, dtype=DType.from_np(arr.dtype))
        x = np.ascontiguousarray(arr.astype(DType.to_np(t.dtype)))
        _lib().sk_tensor_from_cpu(t.handle, x.ctypes.data)
        return t

    def free(self):
        if self._handle:
            _lib().sk_tensor_free(self._handle)
            self._handle = None

    @property
    def handle(self): return self._handle

    @property
    def shape(self):
        if not self._shape and self._handle:
            s = (ctypes.c_uint32 * 4)()
            ndim = _lib().sk_tensor_shape(self._handle, s)
            self._shape = tuple(s[i] for i in range(ndim))
        return self._shape

    @property
    def dtype(self): return self._dtype

    @property
    def numpy(self) -> np.ndarray:
        arr = np.empty(self.shape, dtype=DType.to_np(self.dtype))
        _lib().sk_tensor_to_cpu(self._handle, arr.ctypes.data)
        return arr

    def __repr__(self):
        return f"Tensor({list(self.shape)}, dtype={self.dtype}, handle={hex(self._handle) if self._handle else 0})"

    def __del__(self):
        # Don't auto-free from __del__ — gc timing is unpredictable with Metal.
        # Call .free() explicitly or use context manager.
        pass
