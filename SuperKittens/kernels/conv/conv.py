"""conv.py — ctypes bindings for conv1d."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_conv1d.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_uint32]*4
    _lib.sk_conv1d.restype = ctypes.c_int
    return _lib

def conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
           out: np.ndarray | None = None) -> np.ndarray:
    """Causal depthwise Conv1D. x: (B, L, C), weight: (C, K), bias: (C,)."""
    assert x.dtype == np.float16 and weight.dtype == np.float16 and bias.dtype == np.float16
    B, L, C = x.shape; K = weight.shape[1]
    if out is None: out = np.empty_like(x)
    x, weight, bias, out = map(np.ascontiguousarray, (x, weight, bias, out))
    lib = _load()
    ret = lib.sk_conv1d(x.ctypes.data, weight.ctypes.data, bias.ctypes.data,
                        out.ctypes.data, B, L, C, K)
    if ret: raise RuntimeError(f"sk_conv1d failed: {ret}")
    return out
