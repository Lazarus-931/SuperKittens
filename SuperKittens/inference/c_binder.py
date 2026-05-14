"""Centralized ctypes binder for SK model family dylibs.

Each family declares a verb -> (argtypes, restype) dict and calls bind(family,
abi). The binder opens libsk.dylib (SK_DYLIB env or build/libsk.dylib relative
to repo root), resolves each verb as ``sk_<family>_<verb>``, applies the
signatures, and returns the CDLL handle. Optional symbols are declared by
wrapping the (argtypes, restype) tuple with ``optional(...)``; missing optional
symbols are silently skipped instead of raising.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import asdict
from pathlib import Path


def optional(argtypes, restype):
    """Mark an ABI entry as optional (skipped if the symbol is absent)."""
    return (argtypes, restype, True)


def _default_dylib_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "build" / "libsk.dylib")


_cache: dict[str, ctypes.CDLL] = {}


def bind(family: str, abi: dict) -> ctypes.CDLL:
    """Open libsk.dylib and register ``sk_<family>_<verb>`` for each abi entry.

    abi values are ``(argtypes, restype)`` tuples; use ``optional(...)`` for
    symbols that may be absent in older builds.
    """
    dylib = os.environ.get("SK_DYLIB") or os.environ.get("SK_LIB") or _default_dylib_path()
    lib = _cache.get(dylib)
    if lib is None:
        lib = ctypes.CDLL(dylib)
        _cache[dylib] = lib
    for verb, sig in abi.items():
        sym = f"sk_{family}_{verb}"
        is_optional = len(sig) == 3 and sig[2] is True
        argtypes, restype = sig[0], sig[1]
        if is_optional and not hasattr(lib, sym):
            continue
        fn = getattr(lib, sym)
        fn.argtypes = list(argtypes)
        fn.restype = restype
    return lib


class CtypesConfig:
    """Mixin: copy dataclass fields into a ctypes.Structure via to_c()."""

    def to_c(self, struct_cls):
        cs = struct_cls()
        for k, v in asdict(self).items():
            if hasattr(cs, k):
                setattr(cs, k, v)
        return cs
