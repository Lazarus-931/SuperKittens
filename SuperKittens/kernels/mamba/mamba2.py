"""mamba2.py — ctypes bindings for Mamba-2 kernels."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)

    _lib.sk_mamba2_ssd.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_uint32]*5
    _lib.sk_mamba2_ssd.restype = ctypes.c_int

    _lib.sk_conv1d_silu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_uint32]*3
    _lib.sk_conv1d_silu.restype = ctypes.c_int

    _lib.sk_gate_norm.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_uint32]*3 + [ctypes.c_float]
    _lib.sk_gate_norm.restype = ctypes.c_int
    return _lib

def ssd(Q: np.ndarray, K: np.ndarray, V: np.ndarray, A_log: np.ndarray,
        out: np.ndarray | None = None) -> np.ndarray:
    """Mamba-2 selective scan. Q,K: (B*H, L, Ds). V: (B*H, L, Dv). A_log: (B*H, L)."""
    assert Q.dtype == np.float16 and K.dtype == np.float16 and V.dtype == np.float16
    if A_log.dtype != np.float16: A_log = A_log.astype(np.float16)
    BH, L, Ds = Q.shape; Dv = V.shape[2]
    if out is None: out = np.empty((BH, L, Dv), dtype=np.float16)
    Q,K,V,A_log,out = map(np.ascontiguousarray, (Q,K,V,A_log,out))
    lib = _load()
    ret = lib.sk_mamba2_ssd(Q.ctypes.data, K.ctypes.data, V.ctypes.data,
                            A_log.ctypes.data, out.ctypes.data, 1, L, Ds, Dv, BH)
    if ret: raise RuntimeError(f"mamba2_ssd failed: {ret}")
    return out

def conv1d_silu(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                out: np.ndarray | None = None) -> np.ndarray:
    """Depthwise causal Conv1D + SiLU. x: (B, L, C), weight: (C, 4), bias: (C,)."""
    assert x.dtype == np.float16
    B, L, C = x.shape
    if out is None: out = np.empty_like(x)
    x,w,b,out = map(np.ascontiguousarray, (x,weight,bias,out))
    lib = _load()
    ret = lib.sk_conv1d_silu(x.ctypes.data, w.ctypes.data, b.ctypes.data, out.ctypes.data, B, L, C)
    if ret: raise RuntimeError(f"conv1d_silu failed: {ret}")
    return out

def gate_norm(ssm_out: np.ndarray, z: np.ndarray, weight: np.ndarray,
              out: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    """Gate (silu(z)*ssm_out) + RMSNorm. ssm_out,z: (B, L, E), weight: (E,)."""
    assert ssm_out.dtype == np.float16
    B, L, E = ssm_out.shape
    if out is None: out = np.empty_like(ssm_out)
    ssm_out,z,weight,out = map(np.ascontiguousarray, (ssm_out,z,weight,out))
    lib = _load()
    ret = lib.sk_gate_norm(ssm_out.ctypes.data, z.ctypes.data, weight.ctypes.data,
                           out.ctypes.data, B, L, E, eps)
    if ret: raise RuntimeError(f"gate_norm failed: {ret}")
    return out
