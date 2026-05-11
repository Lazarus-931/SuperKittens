"""flash_attn.py — ctypes binding for the ds4-derived flash_attn_ext_vec."""
from __future__ import annotations
import ctypes, os
from pathlib import Path
import numpy as np

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB",
                           str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_flash_attn_ext_vec.argtypes = (
        [ctypes.c_void_p] * 5
        + [ctypes.c_uint32] * 7
        + [ctypes.c_int, ctypes.c_float, ctypes.c_int32, ctypes.c_int32]
    )
    _lib.sk_flash_attn_ext_vec.restype = ctypes.c_int
    return _lib


def flash_attn_ext_vec(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                       mask: np.ndarray | None = None,
                       *, scale: float | None = None,
                       nsg: int = 4, nwg: int = 1) -> np.ndarray:
    """ds4-derived MLA-shaped flash attention.

    Q   : (B, H,    S_q,  dk) fp32
    K   : (B, H_kv, S_kv, dk) fp16
    V   : (B, H_kv, S_kv, dv) fp16
    mask: (S_q, S_kv) fp16 with -inf for masked positions, or None
    Returns: (B, S_q, H, dv) fp32

    (dk, dv) instantiations available: (128,128), (512,512). Add more by
    appending a one-line template instantiation in flash_attn.metal.
    """
    assert Q.dtype == np.float32 and K.dtype == np.float16 and V.dtype == np.float16
    B, H, S_q, dk = Q.shape
    _, H_kv, S_kv, dk2 = K.shape
    assert dk2 == dk
    _, _, _, dv = V.shape
    if scale is None:
        scale = 1.0 / np.sqrt(dk)
    Q = np.ascontiguousarray(Q); K = np.ascontiguousarray(K); V = np.ascontiguousarray(V)
    out = np.empty((B, S_q, H, dv), dtype=np.float32)
    if mask is not None:
        assert mask.dtype == np.float16 and mask.shape == (S_q, S_kv)
        mask = np.ascontiguousarray(mask)
        m_ptr = mask.ctypes.data
        has_mask = 1
    else:
        m_ptr, has_mask = None, 0
    lib = _load()
    rc = lib.sk_flash_attn_ext_vec(
        Q.ctypes.data, K.ctypes.data, V.ctypes.data, m_ptr, out.ctypes.data,
        B, H, H_kv, S_q, S_kv, dk, dv, has_mask, float(scale), int(nsg), int(nwg))
    if rc:
        raise RuntimeError(f"sk_flash_attn_ext_vec failed: {rc}")
    return out


def causal_mask(S_q: int, S_kv: int) -> np.ndarray:
    """Standard causal mask: -inf above the diagonal aligned to the right."""
    m = np.zeros((S_q, S_kv), dtype=np.float16)
    offset = S_kv - S_q
    for i in range(S_q):
        for j in range(offset + i + 1, S_kv):
            m[i, j] = np.float16(-65504.0)
    return m
