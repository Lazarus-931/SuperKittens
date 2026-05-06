"""rms_norm.py — ctypes binding for RMSNorm."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_rmsnorm.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_uint32]*2 + [ctypes.c_float]
    _lib.sk_rmsnorm.restype = ctypes.c_int
    return _lib

def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5,
            out: np.ndarray | None = None) -> np.ndarray:
    assert x.dtype == np.float16 and weight.dtype == np.float16
    rows, d = x.shape
    assert weight.shape == (d,)
    if out is None: out = np.empty_like(x)
    x, weight, out = map(np.ascontiguousarray, (x, weight, out))
    lib = _load()
    ret = lib.sk_rmsnorm(x.ctypes.data, weight.ctypes.data, out.ctypes.data, rows, d, eps)
    if ret: raise RuntimeError(f"sk_rmsnorm failed: {ret}")
    return out
