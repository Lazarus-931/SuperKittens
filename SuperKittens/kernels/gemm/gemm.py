"""gemm.py — ctypes bindings for GEMM (fp16 + fp8)."""
import ctypes, os, numpy as np
from pathlib import Path

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB", str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    for fn in ("sk_gemm_fp16", "sk_gemm_fp8"):
        getattr(_lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]
        getattr(_lib, fn).restype = ctypes.c_int
    return _lib

def _dispatch(kernel: str, A: np.ndarray, B: np.ndarray,
              out: np.ndarray | None, bias: np.ndarray | None,
              M: int, N: int, K: int, transA: bool, transB: bool) -> np.ndarray:
    ldA = K if not transA else M
    ldB = N if not transB else K
    ldC = N
    if out is None: out = np.empty((M, N), dtype=A.dtype)
    A, B, out = map(np.ascontiguousarray, (A, B, out))
    lib = _load()
    fn = getattr(lib, kernel)
    ret = fn(A.ctypes.data, B.ctypes.data, out.ctypes.data,
             bias.ctypes.data if bias is not None else None,
             M, N, K, ldA, ldB, ldC, int(transA), int(transB), int(bias is not None))
    if ret: raise RuntimeError(f"{kernel} failed: {ret}")
    return out

def gemm_fp16(A: np.ndarray, B: np.ndarray,
              out: np.ndarray | None = None, bias: np.ndarray | None = None,
              transA: bool = False, transB: bool = False) -> np.ndarray:
    M = A.shape[1] if transA else A.shape[0]
    K_inner = A.shape[0] if transA else A.shape[1]
    N = B.shape[0] if transB else B.shape[1]
    return _dispatch("sk_gemm_fp16", A, B, out, bias, M, N, K_inner, transA, transB)

def gemm_fp8(A: np.ndarray, B: np.ndarray,
             out: np.ndarray | None = None, bias: np.ndarray | None = None,
             transA: bool = False, transB: bool = False) -> np.ndarray:
    M = A.shape[1] if transA else A.shape[0]
    K_inner = A.shape[0] if transA else A.shape[1]
    N = B.shape[0] if transB else B.shape[1]
    return _dispatch("sk_gemm_fp8", A, B, out, bias, M, N, K_inner, transA, transB)
