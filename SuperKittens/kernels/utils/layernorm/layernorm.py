"""layernorm.py — ctypes bindings for LayerNorm, RMSNorm."""
import ctypes
import numpy as np
from pathlib import Path
import os

_lib = None

def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[4] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_layernorm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
    _lib.sk_layernorm.restype = ctypes.c_int
    _lib.sk_rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
    _lib.sk_rmsnorm.restype = ctypes.c_int
    return _lib

def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
              out: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    assert x.dtype == np.float16 and gamma.dtype == np.float16 and beta.dtype == np.float16
    rows, d = x.shape
    assert gamma.shape == (d,) and beta.shape == (d,)
    if out is None: out = np.empty_like(x)
    x, gamma, beta, out = map(np.ascontiguousarray, (x, gamma, beta, out))
    lib = _load()
    ret = lib.sk_layernorm(x.ctypes.data, gamma.ctypes.data, beta.ctypes.data, out.ctypes.data, rows, d, eps)
    if ret: raise RuntimeError(f"sk_layernorm failed: {ret}")
    return out

def rmsnorm(x: np.ndarray, weight: np.ndarray,
            out: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    assert x.dtype == np.float16 and weight.dtype == np.float16
    rows, d = x.shape
    assert weight.shape == (d,)
    if out is None: out = np.empty_like(x)
    x, weight, out = map(np.ascontiguousarray, (x, weight, out))
    lib = _load()
    ret = lib.sk_rmsnorm(x.ctypes.data, weight.ctypes.data, out.ctypes.data, rows, d, eps)
    if ret: raise RuntimeError(f"sk_rmsnorm failed: {ret}")
    return out
