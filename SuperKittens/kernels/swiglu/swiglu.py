"""swiglu.py — ctypes bindings for fused SwiGLU."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_swiglu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    _lib.sk_swiglu.restype = ctypes.c_int
    return _lib

def swiglu(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Fused SwiGLU: y = silu(gate) * up. x: (rows, 2*d) where first d=gate, second d=up."""
    assert x.dtype == np.float16 and x.ndim == 2
    rows, d2 = x.shape
    assert d2 % 2 == 0
    if out is None: out = np.empty((rows, d2//2), dtype=np.float16)
    x,out = map(np.ascontiguousarray, (x,out))
    lib = _load()
    ret = lib.sk_swiglu(x.ctypes.data, out.ctypes.data, rows, d2)
    if ret: raise RuntimeError(f"sk_swiglu failed: {ret}")
    return out
