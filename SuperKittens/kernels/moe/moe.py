"""moe.py — ctypes bindings for SuperKittens MoE primitives.

Public functions:
    router(x, W, top_k) → (top_idx, top_score)
    swiglu_pair(x, W_gate, W_up, exp_ids)             — fp16 weights
    swiglu_pair_iq2xxs(x, W_gate, W_up, exp_ids,
                        D, N_int)                      — IQ2_XXS weights (~2 b/wt)
    down_scatter(hidden, W_down, exp_ids,
                  route_w, residual)                   — fp16 weights
    down_scatter_q2k(hidden, W_down, exp_ids,
                      route_w, residual, D, N_int)     — Q2_K weights (~2.625 b/wt)
"""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import numpy as np

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    dylib = os.environ.get(
        "SK_DYLIB",
        str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"),
    )
    _lib = ctypes.CDLL(dylib)

    _lib.sk_moe_router.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 4
    _lib.sk_moe_router.restype  = ctypes.c_int

    _lib.sk_moe_swiglu_pair.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 5
    _lib.sk_moe_swiglu_pair.restype  = ctypes.c_int

    _lib.sk_moe_down_scatter.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 5
    _lib.sk_moe_down_scatter.restype  = ctypes.c_int

    _lib.sk_moe_swiglu_pair_iq2xxs.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 5
    _lib.sk_moe_swiglu_pair_iq2xxs.restype  = ctypes.c_int

    _lib.sk_moe_down_scatter_q2k.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 5
    _lib.sk_moe_down_scatter_q2k.restype  = ctypes.c_int
    return _lib


def swiglu_pair_iq2xxs(x: np.ndarray, W_gate_bytes: bytes, W_up_bytes: bytes,
                       exp_ids: np.ndarray, *, E: int, D: int, N_int: int) -> np.ndarray:
    """IQ2_XXS-weight variant of swiglu_pair.
       x:(T,D) fp16, W_gate/W_up: raw block_iq2_xxs bytes laid out as
       (E, N_int, D/256) blocks (66 bytes each). exp_ids:(T,top_k) i32.
       Returns out:(T,top_k,N_int) fp16."""
    assert x.dtype == np.float16 and exp_ids.dtype == np.int32
    T, _D = x.shape; assert _D == D
    T2, top_k = exp_ids.shape; assert T == T2
    Wg = np.frombuffer(W_gate_bytes, dtype=np.uint8)
    Wu = np.frombuffer(W_up_bytes,   dtype=np.uint8)
    expected = E * N_int * (D // 256) * 66
    assert Wg.nbytes == expected and Wu.nbytes == expected, f"IQ2_XXS bytes mismatch (want {expected})"
    x = np.ascontiguousarray(x); exp_ids = np.ascontiguousarray(exp_ids)
    out = np.zeros((T, top_k, N_int), dtype=np.float16)
    lib = _load()
    r = lib.sk_moe_swiglu_pair_iq2xxs(x.ctypes.data, Wg.ctypes.data, Wu.ctypes.data,
                                       exp_ids.ctypes.data, out.ctypes.data,
                                       T, top_k, E, D, N_int)
    if r: raise RuntimeError(f"sk_moe_swiglu_pair_iq2xxs failed: {r}")
    return out


def down_scatter_q2k(hidden: np.ndarray, W_down_bytes: bytes, exp_ids: np.ndarray,
                     route_w: np.ndarray, residual: np.ndarray, *, E: int, D: int) -> np.ndarray:
    """Q2_K-weight variant of down_scatter.
       hidden:(T,top_k,N_int) fp16. W_down: raw block_q2_K bytes as
       (E, D, N_int/256) blocks (84 bytes each). exp_ids:(T,top_k) i32.
       route_w:(T,top_k) fp16, residual:(T,D) fp16. Returns out:(T,D) fp16."""
    assert hidden.dtype == np.float16 and exp_ids.dtype == np.int32
    assert route_w.dtype == np.float16 and residual.dtype == np.float16
    T, top_k, N_int = hidden.shape
    assert residual.shape == (T, D)
    Wd = np.frombuffer(W_down_bytes, dtype=np.uint8)
    expected = E * D * (N_int // 256) * 84
    assert Wd.nbytes == expected, f"Q2_K bytes mismatch (want {expected})"
    hidden = np.ascontiguousarray(hidden); exp_ids = np.ascontiguousarray(exp_ids)
    route_w = np.ascontiguousarray(route_w); residual = np.ascontiguousarray(residual)
    out = np.zeros((T, D), dtype=np.float16)
    lib = _load()
    r = lib.sk_moe_down_scatter_q2k(hidden.ctypes.data, Wd.ctypes.data,
                                     exp_ids.ctypes.data, route_w.ctypes.data,
                                     residual.ctypes.data, out.ctypes.data,
                                     T, top_k, E, D, N_int)
    if r: raise RuntimeError(f"sk_moe_down_scatter_q2k failed: {r}")
    return out


def swiglu_pair(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray,
                exp_ids: np.ndarray) -> np.ndarray:
    """Fused gate + up + SiLU + mul (per-expert matvec).
       x:(T,D), W_gate:(E,N_int,D), W_up:(E,N_int,D), exp_ids:(T,top_k) i32
       -> out:(T,top_k,N_int) fp16."""
    assert x.dtype == np.float16 and W_gate.dtype == np.float16 and W_up.dtype == np.float16
    assert exp_ids.dtype == np.int32
    T, D = x.shape
    E, N_int, D2 = W_gate.shape
    assert D == D2 and W_up.shape == W_gate.shape
    T2, top_k = exp_ids.shape
    assert T == T2
    x       = np.ascontiguousarray(x)
    W_gate  = np.ascontiguousarray(W_gate)
    W_up    = np.ascontiguousarray(W_up)
    exp_ids = np.ascontiguousarray(exp_ids)
    out     = np.zeros((T, top_k, N_int), dtype=np.float16)
    lib = _load()
    r = lib.sk_moe_swiglu_pair(x.ctypes.data, W_gate.ctypes.data, W_up.ctypes.data,
                                exp_ids.ctypes.data, out.ctypes.data,
                                T, top_k, E, D, N_int)
    if r: raise RuntimeError(f"sk_moe_swiglu_pair failed: {r}")
    return out


def down_scatter(hidden: np.ndarray, W_down: np.ndarray, exp_ids: np.ndarray,
                 route_w: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Fused down-proj + routing-weight scale + residual add.
       hidden:(T,top_k,N_int), W_down:(E,D,N_int), exp_ids:(T,top_k) i32,
       route_w:(T,top_k) fp16, residual:(T,D) fp16
       -> out:(T,D) fp16."""
    assert hidden.dtype == np.float16 and W_down.dtype == np.float16
    assert exp_ids.dtype == np.int32
    assert route_w.dtype == np.float16 and residual.dtype == np.float16
    T, top_k, N_int = hidden.shape
    E, D, N_int2 = W_down.shape
    assert N_int == N_int2 and exp_ids.shape == (T, top_k)
    assert route_w.shape == (T, top_k) and residual.shape == (T, D)
    hidden   = np.ascontiguousarray(hidden)
    W_down   = np.ascontiguousarray(W_down)
    exp_ids  = np.ascontiguousarray(exp_ids)
    route_w  = np.ascontiguousarray(route_w)
    residual = np.ascontiguousarray(residual)
    out      = np.zeros((T, D), dtype=np.float16)
    lib = _load()
    r = lib.sk_moe_down_scatter(hidden.ctypes.data, W_down.ctypes.data,
                                 exp_ids.ctypes.data, route_w.ctypes.data,
                                 residual.ctypes.data, out.ctypes.data,
                                 T, top_k, E, D, N_int)
    if r: raise RuntimeError(f"sk_moe_down_scatter failed: {r}")
    return out


def router(x: np.ndarray, W: np.ndarray, top_k: int):
    """x:(T,D) fp16, W:(D,N) fp16 -> (top_idx (T,K) i32, top_score (T,K) fp16)."""
    assert x.dtype == np.float16 and W.dtype == np.float16
    T, D = x.shape
    D2, N = W.shape
    assert D == D2
    x = np.ascontiguousarray(x)
    W = np.ascontiguousarray(W)
    top_idx   = np.zeros((T, top_k), dtype=np.int32)
    top_score = np.zeros((T, top_k), dtype=np.float16)
    lib = _load()
    r = lib.sk_moe_router(x.ctypes.data, W.ctypes.data,
                          top_idx.ctypes.data, top_score.ctypes.data,
                          T, D, N, top_k)
    if r: raise RuntimeError(f"sk_moe_router failed: {r}")
    return top_idx, top_score
