"""rotary.py — ctypes bindings for RoPE."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_rope.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    _lib.sk_rope.restype = ctypes.c_int
    return _lib

def rope(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply RoPE in-place. Q,K: (n_heads, seq, head_dim). cos,sin: (seq, head_dim/2)."""
    assert q.dtype == np.float16 and k.dtype == np.float16
    n_heads, seq, head_dim = q.shape
    assert k.shape == q.shape
    assert cos.shape == (seq, head_dim // 2) and sin.shape == (seq, head_dim // 2)
    q, k = map(np.ascontiguousarray, (q, k))
    lib = _load()
    ret = lib.sk_rope(q.ctypes.data, k.ctypes.data,
                       np.ascontiguousarray(cos).ctypes.data,
                       np.ascontiguousarray(sin).ctypes.data,
                       seq, head_dim, n_heads)
    if ret: raise RuntimeError(f"sk_rope failed: {ret}")
    return q, k
