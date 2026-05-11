"""embedding.py — ctypes binding for token-id embedding lookup."""
import ctypes, os
from pathlib import Path

import numpy as np

_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB",
                           str(Path(__file__).resolve().parents[4] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_embedding_lookup.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ]
    _lib.sk_embedding_lookup.restype = ctypes.c_int
    return _lib


def embedding_lookup(table: np.ndarray, ids: np.ndarray,
                     out: np.ndarray | None = None) -> np.ndarray:
    """Gather rows from `table` (V, D) fp16 by `ids` (N,) int32.

    Out-of-range ids zero-fill (won't crash). D must be divisible by 4.
    """
    assert table.dtype == np.float16, f"table must be fp16, got {table.dtype}"
    assert ids.dtype in (np.int32, np.int64), f"ids must be int32/int64, got {ids.dtype}"
    if ids.dtype == np.int64:
        ids = ids.astype(np.int32)
    assert table.ndim == 2 and ids.ndim == 1
    V, D = table.shape
    assert D % 4 == 0, f"D ({D}) must be divisible by 4"
    N = ids.shape[0]

    if out is None:
        out = np.empty((N, D), dtype=np.float16)
    else:
        assert out.shape == (N, D) and out.dtype == np.float16

    table = np.ascontiguousarray(table)
    ids   = np.ascontiguousarray(ids)
    out   = np.ascontiguousarray(out)

    rc = _load().sk_embedding_lookup(
        table.ctypes.data, ids.ctypes.data, out.ctypes.data, N, D, V,
    )
    if rc != 0:
        raise RuntimeError(f"sk_embedding_lookup failed: rc={rc}")
    return out
