"""Core: dylib loading, device singleton, buffer helpers."""
import ctypes, os

_device = None

class Device:
    def __init__(self, dylib_path=None):
        path = dylib_path or os.environ.get("SK_DYLIB", "build/libsk.dylib")
        self.lib = ctypes.CDLL(path)
        self._declare()

    def _declare(self):
        # activation
        self.lib.sk_activation.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32
        ]
        self.lib.sk_activation.restype = ctypes.c_int

def get_device() -> Device:
    global _device
    if _device is None:
        _device = Device()
    return _device

def to_metal(x):
    """Ensure fp16, contiguous. Returns (array, ptr)."""
    import numpy as np
    if x.dtype != np.float16:
        x = x.astype(np.float16)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x
