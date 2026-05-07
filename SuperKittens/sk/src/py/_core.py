"""Core: dylib loading, device singleton, numpy helpers."""
import ctypes, os, numpy as np

_dylib = None

def _load():
    global _dylib
    if _dylib is not None:
        return _dylib
    path = os.environ.get("SK_DYLIB", "build/libsk.dylib")
    _dylib = ctypes.CDLL(path)
    _declare_all()
    return _dylib

def _declare_all():
    lib = _dylib

    # Activation
    lib.sk_activation.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_uint32, ctypes.c_uint32]
    lib.sk_activation.restype  = ctypes.c_int

    # Attention
    lib.sk_attn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.c_int]
    lib.sk_attn.restype  = ctypes.c_int

    # RoPE
    lib.sk_rope.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.sk_rope.restype  = ctypes.c_int

    # RMSNorm
    lib.sk_rmsnorm.argtypes = [ctypes.c_char_p,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
    lib.sk_rmsnorm.restype  = ctypes.c_int

    # Conv1d
    lib.sk_conv1d.argtypes = [ctypes.c_char_p,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.sk_conv1d.restype = ctypes.c_int

    # GEMM fp16
    lib.sk_gemm_fp16.argtypes = [ctypes.c_char_p,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.sk_gemm_fp16.restype = ctypes.c_int

    # Gate norm
    lib.sk_gate_norm.argtypes = [ctypes.c_char_p,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
    lib.sk_gate_norm.restype = ctypes.c_int

    # SwiGLU
    lib.sk_swiglu.argtypes = [ctypes.c_char_p,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_uint32, ctypes.c_uint32]
    lib.sk_swiglu.restype = ctypes.c_int

def _as_fp16(x: np.ndarray) -> np.ndarray:
    """Ensure fp16 and contiguous."""
    if x.dtype != np.float16:
        x = x.astype(np.float16)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x
