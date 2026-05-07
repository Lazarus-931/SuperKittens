"""paged_attn.py — ctypes bindings for paged attention."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_paged_attn.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_uint32]*6
    _lib.sk_paged_attn.restype = ctypes.c_int
    return _lib

def paged_attention(Q: np.ndarray, K_cache: np.ndarray, V_cache: np.ndarray,
                    block_table: np.ndarray, seq_lens: np.ndarray,
                    block_size: int = 16,
                    out: np.ndarray | None = None) -> np.ndarray:
    """Paged attention decode. Q: (num_seqs, num_heads, head_dim).
    K_cache, V_cache: (num_blocks, block_size, num_kv_heads, head_dim).
    block_table: (num_seqs, max_blocks) int32. seq_lens: (num_seqs,) int32."""
    assert Q.dtype == np.float16
    num_seqs, num_heads, head_dim = Q.shape
    num_kv_heads = K_cache.shape[2]
    _, blk_sz, _, _ = K_cache.shape
    max_blocks = block_table.shape[1]
    if out is None: out = np.empty_like(Q)
    Q,K_cache,V_cache,block_table,seq_lens,out = map(np.ascontiguousarray, (Q,K_cache,V_cache,block_table,seq_lens,out))
    lib = _load()
    ret = lib.sk_paged_attn(Q.ctypes.data, K_cache.ctypes.data, V_cache.ctypes.data,
                            block_table.ctypes.data, seq_lens.ctypes.data, out.ctypes.data,
                            num_seqs, num_heads, head_dim, num_kv_heads, blk_sz, max_blocks)
    if ret: raise RuntimeError(f"sk_paged_attn failed: {ret}")
    return out
