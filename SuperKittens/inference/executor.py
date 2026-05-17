"""Executor — Python handle around the C ABI in executor/executor_c.h.

The Executor wraps a model's Metal command queue, PSO cache, scratch pool,
residency, and ICB lifecycle. This file is pure ctypes plumbing; no Metal
calls happen in Python.
"""
from __future__ import annotations

import ctypes
from typing import Optional, Sequence

from SuperKittens.inference.c_binder import bind


class SKMtlSize(ctypes.Structure):
    """Mirror of sk_mtl_size_t. ctypes passes by value into the C ABI."""
    _fields_ = [
        ("width",  ctypes.c_uint64),
        ("height", ctypes.c_uint64),
        ("depth",  ctypes.c_uint64),
    ]


# (argtypes, restype) per verb. Handles are c_void_p everywhere — never int —
# so pointers don't get truncated on arm64.
_ABI = {
    "default_device":            (( ),                                                                                   ctypes.c_void_p),
    "create":                    (( ctypes.c_void_p, ),                                                                  ctypes.c_void_p),
    "destroy":                   (( ctypes.c_void_p, ),                                                                  None),
    "dispatch":                  (( ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                    SKMtlSize, SKMtlSize ),                                                              None),
    "record_token_icb_begin":    (( ctypes.c_void_p, ),                                                                  None),
    "record_token_icb_end":      (( ctypes.c_void_p, ),                                                                  None),
    "replay_token":              (( ctypes.c_void_p, ctypes.c_void_p ),                                                  None),
    "reset":                     (( ctypes.c_void_p, ),                                                                  None),
    "pso":                       (( ctypes.c_void_p, ctypes.c_char_p ),                                                  ctypes.c_void_p),
    "scratch":                   (( ctypes.c_void_p, ctypes.c_uint64 ),                                                  ctypes.c_void_p),
}


_lib = None
def _lib_handle():
    global _lib
    if _lib is None:
        _lib = bind("executor", _ABI)
    return _lib


def _mtl_size(t) -> SKMtlSize:
    # Accept (w, h, d) tuples; default missing dims to 1 (Metal convention).
    if isinstance(t, SKMtlSize):
        return t
    w = int(t[0]); h = int(t[1]) if len(t) > 1 else 1; d = int(t[2]) if len(t) > 2 else 1
    return SKMtlSize(w, h, d)


class Executor:
    """Python handle to a native sk::Executor."""

    def __init__(self, device: int = 0):
        lib = _lib_handle()
        self._lib = lib
        # device is a raw void* (Python int) into MTL::Device. 0 is allowed —
        # the native side degrades to inert (no queue / no lib) for ABI tests.
        self._h = lib.sk_executor_create(ctypes.c_void_p(device))

    # ---- PSO + scratch (test seams) ----------------------------------
    def pso(self, host_name: str) -> int:
        return self._lib.sk_executor_pso(self._h, host_name.encode("utf-8")) or 0

    def scratch(self, nbytes: int) -> int:
        return self._lib.sk_executor_scratch(self._h, ctypes.c_uint64(int(nbytes))) or 0

    # ---- Dispatch ----------------------------------------------------
    def dispatch(self, host_name: str, buffers: Sequence[int],
                 grid, tg) -> None:
        n = len(buffers)
        arr_t = ctypes.c_void_p * n
        arr = arr_t(*[ctypes.c_void_p(b) for b in buffers])
        self._lib.sk_executor_dispatch(
            self._h, host_name.encode("utf-8"),
            arr, ctypes.c_int(n),
            _mtl_size(grid), _mtl_size(tg))

    # ---- ICB lifecycle ----------------------------------------------
    def record_token_icb_begin(self) -> None:
        self._lib.sk_executor_record_token_icb_begin(self._h)

    def record_token_icb_end(self) -> None:
        self._lib.sk_executor_record_token_icb_end(self._h)

    def replay_token(self, cmd_buffer: int) -> None:
        self._lib.sk_executor_replay_token(self._h, ctypes.c_void_p(cmd_buffer))

    def reset(self) -> None:
        self._lib.sk_executor_reset(self._h)

    def close(self) -> None:
        h = getattr(self, "_h", None)
        if h:
            self._lib.sk_executor_destroy(h)
            self._h = None

    def __del__(self):
        try: self.close()
        except Exception: pass


def default_device() -> int:
    """Return the libsk default MTLDevice as a raw pointer (0 if unavailable)."""
    return _lib_handle().sk_executor_default_device() or 0


__all__ = ["Executor", "SKMtlSize", "default_device"]
