"""Sampler — Python handle around the C ABI in sampling/sampler_c.h.

A Sampler is a stateful policy object that records sampling dispatches into a
caller-owned MTL::ComputeCommandEncoder. Model launchers ask the Sampler to
emit work for each decoded token; this file is pure plumbing — no Metal calls
happen in pure Python, only ctypes-level wiring.
"""
from __future__ import annotations

import ctypes
from typing import Optional

from SuperKittens.inference.c_binder import bind, optional


# All sampler entry points. Handles are opaque void* — ctypes.c_void_p
# everywhere so we never accidentally truncate a pointer on arm64.
_ABI = {
    "create":           (( ctypes.c_void_p, ctypes.c_void_p ), ctypes.c_void_p),
    "set_greedy":       (( ctypes.c_void_p, ),                 None),
    "set_top_p":        (( ctypes.c_void_p, ctypes.c_float, ctypes.c_float ), None),
    "set_min_p":        (( ctypes.c_void_p, ctypes.c_float, ctypes.c_float ), None),
    "set_multinomial":  (( ctypes.c_void_p, ctypes.c_float ),   None),
    "set_seed":         (( ctypes.c_void_p, ctypes.c_uint64 ),  None),
    "sample":           (( ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_uint32, ctypes.c_void_p ),  None),
    "destroy":          (( ctypes.c_void_p, ),                  None),
}


_lib = None
def _lib_handle():
    """Lazily bind the dylib. Tests stub this attribute to avoid loading."""
    global _lib
    if _lib is None:
        _lib = bind("sampler", _ABI)
    return _lib


class Sampler:
    """Composable sampling policy. Construct via class methods.

    The returned object holds an opaque C handle. Pass it to a model launcher
    that knows how to feed the active MTL encoder into ``_record_into``.
    """

    # Mirror of the policy on the Python side so generate() can still execute
    # a CPU fallback while launcher integration lands. The C handle remains
    # authoritative for the GPU path.
    mode: str = "greedy"
    temp: float = 1.0
    p: float = 1.0
    seed: int = 0

    def __init__(self, device: int = 0, queue: int = 0):
        # device/queue are raw void* (Python ints) into MTL::Device /
        # MTL::CommandQueue from the model's launcher. Tests pass 0/0 — the
        # native side is allowed to construct with null device for ABI tests
        # via the test seam at the bottom of this file.
        lib = _lib_handle()
        self._lib = lib
        self._h = lib.sk_sampler_create(ctypes.c_void_p(device),
                                         ctypes.c_void_p(queue))

    # ----- class constructors -----------------------------------------
    @classmethod
    def greedy(cls, device: int = 0, queue: int = 0) -> "Sampler":
        s = cls(device, queue); s._lib.sk_sampler_set_greedy(s._h)
        s.mode = "greedy"; return s

    @classmethod
    def top_p(cls, p: float, temp: float, device: int = 0, queue: int = 0) -> "Sampler":
        s = cls(device, queue)
        s._lib.sk_sampler_set_top_p(s._h, ctypes.c_float(p), ctypes.c_float(temp))
        s.mode = "top_p"; s.p = float(p); s.temp = float(temp); return s

    @classmethod
    def min_p(cls, p: float, temp: float, device: int = 0, queue: int = 0) -> "Sampler":
        s = cls(device, queue)
        s._lib.sk_sampler_set_min_p(s._h, ctypes.c_float(p), ctypes.c_float(temp))
        s.mode = "min_p"; s.p = float(p); s.temp = float(temp); return s

    @classmethod
    def multinomial(cls, temp: float, device: int = 0, queue: int = 0) -> "Sampler":
        s = cls(device, queue)
        s._lib.sk_sampler_set_multinomial(s._h, ctypes.c_float(temp))
        s.mode = "multinomial"; s.temp = float(temp); return s

    # ----- mutators ---------------------------------------------------
    def set_seed(self, seed: int) -> "Sampler":
        self._lib.sk_sampler_set_seed(self._h, ctypes.c_uint64(seed))
        self.seed = int(seed); return self

    def _record_into(self, logits_buf: int, out_buf: int,
                     vocab_size: int, encoder: int) -> None:
        """Record sampling dispatches into a caller-owned encoder pointer."""
        self._lib.sk_sampler_sample(
            self._h, ctypes.c_void_p(logits_buf), ctypes.c_void_p(out_buf),
            ctypes.c_uint32(vocab_size), ctypes.c_void_p(encoder))

    # ----- lifecycle --------------------------------------------------
    def close(self) -> None:
        h = getattr(self, "_h", None)
        if h:
            self._lib.sk_sampler_destroy(h)
            self._h = None

    def __del__(self):
        try: self.close()
        except Exception: pass


__all__ = ["Sampler"]
