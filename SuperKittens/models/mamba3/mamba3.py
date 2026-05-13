"""mamba3.py — ctypes bindings for Mamba-3 kernels."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)

    _lib.sk_mamba3_pre_ssm.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_uint32]*3 + [ctypes.c_float]
    _lib.sk_mamba3_pre_ssm.restype = ctypes.c_int

    _lib.sk_mamba3_ssm.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_uint32]*5
    _lib.sk_mamba3_ssm.restype = ctypes.c_int

    _lib.sk_mamba3_ssm_state.argtypes = [ctypes.c_void_p]*11 + [ctypes.c_uint32]*5
    _lib.sk_mamba3_ssm_state.restype = ctypes.c_int

    _lib.sk_mamba3_step.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_uint32]*3
    _lib.sk_mamba3_step.restype = ctypes.c_int

    _lib.sk_mamba3_post_ssm.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_uint32]*3 + [ctypes.c_float]
    _lib.sk_mamba3_post_ssm.restype = ctypes.c_int
    return _lib

def pre_ssm(xBC: np.ndarray, dt: np.ndarray, angle: np.ndarray, norm_w: np.ndarray,
            eps: float = 1e-5) -> tuple:
    """Pre-SSM: norm + rotary. xBC: (BH, L, 2*DQ), dt: (BH, L), angle: (BH, L, DQ/2), norm_w: (DQ,).
    Returns: Q, K, V, A, B."""
    assert xBC.dtype == np.float16
    BH, L, DQ2 = xBC.shape; DQ = DQ2 // 2
    Q = np.empty((BH, L, DQ), dtype=np.float16); K = np.empty_like(Q); V = np.empty_like(Q)
    A = np.empty((BH, L), dtype=np.float16); B_out = np.empty((BH, L, DQ), dtype=np.float16)
    xBC,dt,angle,norm_w,Q,K,V,A,B_out = map(np.ascontiguousarray, (xBC,dt,angle,norm_w,Q,K,V,A,B_out))
    lib = _load()
    ret = lib.sk_mamba3_pre_ssm(
        xBC.ctypes.data, dt.ctypes.data, angle.ctypes.data, norm_w.ctypes.data,
        Q.ctypes.data, K.ctypes.data, V.ctypes.data, A.ctypes.data, B_out.ctypes.data,
        BH, L, DQ, eps)
    if ret: raise RuntimeError(f"mamba3_pre_ssm failed: {ret}")
    return Q, K, V, A, B_out

def ssm(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        A: np.ndarray, B: np.ndarray, angle: np.ndarray,
        CS: int = 32, out: np.ndarray | None = None) -> np.ndarray:
    """Mamba-3 selective scan. Q,K: (BH, L, DQ). V: (BH, L, DV). A,B: (BH, L). angle: (BH, L, DQ/2)."""
    assert Q.dtype == np.float16
    BH, L, DQ = Q.shape; DV = V.shape[2]
    if out is None: out = np.empty((BH, L, DV), dtype=np.float16)
    Q,K,V,A,B,angle,out = map(np.ascontiguousarray, (Q,K,V,A,B,angle,out))
    lib = _load()
    ret = lib.sk_mamba3_ssm(Q.ctypes.data, K.ctypes.data, V.ctypes.data,
                            A.ctypes.data, B.ctypes.data, angle.ctypes.data,
                            out.ctypes.data, BH, L, DQ, DV, CS)
    if ret: raise RuntimeError(f"mamba3_ssm failed: {ret}")
    return out

def ssm_state(Q, K, V, A, B, angle, h_state_in=None, h_state_out=None,
              a_cs_in=None, a_cs_out=None, CS=32, out=None):
    """Mamba-3 SSM with optional persistent state.
    h_state: (BH,DQ,DV) fp32   a_cs: (BH,) fp32 running cumsum offset."""
    BH, L, DQ = Q.shape; DV = V.shape[2]
    if out is None: out = np.empty((BH, L, DV), dtype=np.float16)
    Q,K,V,A,B,angle,out = map(np.ascontiguousarray, (Q,K,V,A,B,angle,out))
    hi = h_state_in.ctypes.data if h_state_in is not None else 0
    ho = h_state_out.ctypes.data if h_state_out is not None else 0
    ci = a_cs_in.ctypes.data    if a_cs_in is not None else 0
    co = a_cs_out.ctypes.data   if a_cs_out is not None else 0
    lib = _load()
    ret = lib.sk_mamba3_ssm_state(Q.ctypes.data, K.ctypes.data, V.ctypes.data,
                                  A.ctypes.data, B.ctypes.data, angle.ctypes.data,
                                  out.ctypes.data, hi, ho, ci, co, BH, L, DQ, DV, CS)
    if ret: raise RuntimeError(f"mamba3_ssm_state failed: {ret}")
    return out

def step(q_t, k_t, v_t, a_t, b_t, angle_t, h_state, a_cs, out=None):
    """Mamba-3 decode step. h_state (BH,DQ,DV) fp32 and a_cs (BH,) fp32 modified in place."""
    BH, DQ = q_t.shape; DV = v_t.shape[1]
    assert h_state.dtype == np.float32 and a_cs.dtype == np.float32
    if out is None: out = np.empty((BH, DV), dtype=np.float16)
    q_t,k_t,v_t,a_t,b_t,angle_t,h_state,a_cs,out = map(np.ascontiguousarray,
        (q_t,k_t,v_t,a_t,b_t,angle_t,h_state,a_cs,out))
    lib = _load()
    ret = lib.sk_mamba3_step(q_t.ctypes.data, k_t.ctypes.data, v_t.ctypes.data,
                             a_t.ctypes.data, b_t.ctypes.data, angle_t.ctypes.data,
                             h_state.ctypes.data, a_cs.ctypes.data, out.ctypes.data,
                             BH, DQ, DV)
    if ret: raise RuntimeError(f"mamba3_step failed: {ret}")
    return out

def post_ssm(z: np.ndarray, ssm_out: np.ndarray, norm_w: np.ndarray,
             out: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    """Post-SSM: silu(z)*ssm_out + RMSNorm. z,ssm_out: (BH, L, DV), norm_w: (DV,)."""
    assert z.dtype == np.float16
    BH, L, DV = z.shape
    if out is None: out = np.empty_like(z)
    z,ssm_out,norm_w,out = map(np.ascontiguousarray, (z,ssm_out,norm_w,out))
    lib = _load()
    ret = lib.sk_mamba3_post_ssm(z.ctypes.data, ssm_out.ctypes.data, norm_w.ctypes.data,
                                 out.ctypes.data, BH, L, DV, eps)
    if ret: raise RuntimeError(f"mamba3_post_ssm failed: {ret}")
    return out
