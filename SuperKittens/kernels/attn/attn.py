"""attn.py — ctypes bindings for attention (d=64 FA, d=128 MHA)."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_attn.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_uint32]*4 + [ctypes.c_int]
    _lib.sk_attn.restype = ctypes.c_int
    return _lib

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
              out: np.ndarray | None = None, causal: bool = True) -> np.ndarray:
    """Multi-head / grouped-query attention. Q: (nheads, seq, head_dim), K,V: (n_kv_heads, seq, head_dim)."""
    assert Q.dtype == np.float16
    nheads, seq, head_dim = Q.shape
    n_kv_heads = K.shape[0]
    assert K.shape == (n_kv_heads, seq, head_dim) and V.shape == (n_kv_heads, seq, head_dim)
    assert nheads % n_kv_heads == 0, f"nheads ({nheads}) must be multiple of n_kv_heads ({n_kv_heads})"
    if out is None: out = np.empty_like(Q)
    Q,K,V,out = map(np.ascontiguousarray, (Q,K,V,out))
    lib = _load()
    ret = lib.sk_attn(Q.ctypes.data, K.ctypes.data, V.ctypes.data, out.ctypes.data,
                      seq, head_dim, nheads, n_kv_heads, int(causal))
    if ret: raise RuntimeError(f"sk_attn failed: {ret}")
    return out
