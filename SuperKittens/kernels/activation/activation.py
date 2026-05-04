"""activation.py — ctypes bindings for GELU, SiLU, ReLU."""
import ctypes
import numpy as np
from pathlib import Path

_lib = None

def _load():
    global _lib
    if _lib is not None:
        return _lib
    import os
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_activation.argtypes = [ctypes.c_char_p,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_uint32, ctypes.c_uint32]
    _lib.sk_activation.restype = ctypes.c_int
    return _lib

def _dispatch(name: str, x: np.ndarray, y: np.ndarray):
    assert x.dtype == np.float16 and y.dtype == np.float16
    assert x.shape == y.shape
    rows, cols = x.shape
    lib = _load()
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    ret = lib.sk_activation(name.encode(),
                            x.ctypes.data, y.ctypes.data,
                            rows, cols)
    if ret != 0:
        raise RuntimeError(f"sk_activation({name}) failed with code {ret}")

def gelu(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if out is None: out = np.empty_like(x)
    _dispatch("gelu", x, out)
    return out

def silu(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if out is None: out = np.empty_like(x)
    _dispatch("silu", x, out)
    return out

def relu(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if out is None: out = np.empty_like(x)
    _dispatch("relu", x, out)
    return out
