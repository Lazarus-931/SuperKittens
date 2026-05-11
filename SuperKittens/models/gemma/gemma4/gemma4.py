"""gemma4.py — ctypes wrapper around the Gemma 4 launcher (libsk.dylib)."""
import ctypes, os
import numpy as np
from pathlib import Path
from dataclasses import dataclass


class _Config(ctypes.Structure):
    _fields_ = [
        ("batch",              ctypes.c_uint32),
        ("seq_max",            ctypes.c_uint32),
        ("cache_max",          ctypes.c_uint32),
        ("n_layers",           ctypes.c_uint32),
        ("local_period",       ctypes.c_uint32),
        ("d_model",            ctypes.c_uint32),
        ("n_int",              ctypes.c_uint32),
        ("n_heads",            ctypes.c_uint32),
        ("n_kv_heads_local",   ctypes.c_uint32),
        ("n_kv_heads_global",  ctypes.c_uint32),
        ("head_dim_local",     ctypes.c_uint32),
        ("head_dim_global",    ctypes.c_uint32),
        ("window",             ctypes.c_uint32),
        ("prope_p_pairs",      ctypes.c_uint32),
        ("vocab_size",         ctypes.c_uint32),
        ("has_ple",            ctypes.c_int),
        ("eps",                ctypes.c_float),
    ]


class _Weights(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "w_embed", "w_ple",
        "w_pre_attn_norm", "w_post_attn_norm",
        "w_pre_mlp_norm", "w_post_mlp_norm", "w_final_norm",
        "w_qkv", "w_out", "gamma_q", "gamma_k",
        "w_gate", "w_up", "w_down",
        "cos_local", "sin_local", "cos_global", "sin_global",
    )]


_lib = None
def _load():
    global _lib
    if _lib is not None:
        return _lib
    dylib = os.environ.get(
        "SK_DYLIB",
        str(Path(__file__).resolve().parents[4] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)

    _lib.sk_gemma4_create.argtypes = [ctypes.POINTER(_Config)]
    _lib.sk_gemma4_create.restype  = ctypes.c_void_p

    _lib.sk_gemma4_load_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Weights)]
    _lib.sk_gemma4_load_weights.restype  = ctypes.c_int

    _lib.sk_gemma4_forward.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
    _lib.sk_gemma4_forward.restype = ctypes.c_int

    _lib.sk_gemma4_reset.argtypes  = [ctypes.c_void_p]
    _lib.sk_gemma4_reset.restype   = None

    _lib.sk_gemma4_destroy.argtypes = [ctypes.c_void_p]
    _lib.sk_gemma4_destroy.restype  = None
    return _lib


@dataclass
class Gemma4Config:
    n_layers: int
    local_period: int
    d_model: int
    n_int: int
    n_heads: int
    n_kv_heads_local: int
    n_kv_heads_global: int
    head_dim_local: int = 256
    head_dim_global: int = 512
    window: int = 4096
    prope_p_pairs: int = 64
    vocab_size: int = 262144
    has_ple: bool = False
    eps: float = 1e-6
    batch: int = 1
    seq_max: int = 8192
    cache_max: int = 8192


def _preset(name: str) -> Gemma4Config:
    n = name.lower()
    if n in ("e2b",):
        return Gemma4Config(n_layers=35, local_period=6, d_model=1536, n_int=6144,
                            n_heads=4, n_kv_heads_local=4, n_kv_heads_global=1,
                            window=4096, has_ple=True)
    if n in ("e4b",):
        return Gemma4Config(n_layers=35, local_period=6, d_model=2048, n_int=8192,
                            n_heads=8, n_kv_heads_local=8, n_kv_heads_global=2,
                            window=4096, has_ple=True)
    if n in ("26b", "26b-a4b"):
        return Gemma4Config(n_layers=60, local_period=6, d_model=4608, n_int=12288,
                            n_heads=16, n_kv_heads_local=16, n_kv_heads_global=4,
                            window=1024, has_ple=False)
    if n in ("31b",):
        return Gemma4Config(n_layers=60, local_period=6, d_model=21504, n_int=86016,
                            n_heads=32, n_kv_heads_local=32, n_kv_heads_global=16,
                            window=1024, has_ple=False)
    raise ValueError(f"unknown gemma4 variant: {name}")


def _to_cstruct(c: Gemma4Config) -> _Config:
    cs = _Config()
    cs.batch              = c.batch
    cs.seq_max            = c.seq_max
    cs.cache_max          = c.cache_max
    cs.n_layers           = c.n_layers
    cs.local_period       = c.local_period
    cs.d_model            = c.d_model
    cs.n_int              = c.n_int
    cs.n_heads            = c.n_heads
    cs.n_kv_heads_local   = c.n_kv_heads_local
    cs.n_kv_heads_global  = c.n_kv_heads_global
    cs.head_dim_local     = c.head_dim_local
    cs.head_dim_global    = c.head_dim_global
    cs.window             = c.window
    cs.prope_p_pairs      = c.prope_p_pairs
    cs.vocab_size         = c.vocab_size
    cs.has_ple            = 1 if c.has_ple else 0
    cs.eps                = c.eps
    return cs


class Gemma4:
    """Stateful Gemma 4 inference handle. Holds KV cache between forwards."""

    def __init__(self, variant_or_config):
        cfg = _preset(variant_or_config) if isinstance(variant_or_config, str) else variant_or_config
        self.cfg = cfg
        self._cstruct = _to_cstruct(cfg)
        lib = _load()
        self._h = lib.sk_gemma4_create(ctypes.byref(self._cstruct))
        if not self._h:
            raise RuntimeError("sk_gemma4_create failed")
        self._w_keep = None  # keep np arrays alive while loaded

    def load_weights(self, **arrays: np.ndarray):
        """Pass each weight as a contiguous fp16 numpy array, keyed by field name
        (w_embed, w_qkv, ...). w_ple may be omitted if has_ple=False."""
        keep = []
        w = _Weights()
        for entry in _Weights._fields_:
            f = entry[0]
            a = arrays.get(f)
            if a is None:
                if f == "w_ple" and not self.cfg.has_ple:
                    setattr(w, f, 0)
                    continue
                raise ValueError(f"missing weight: {f}")
            a = np.ascontiguousarray(a, dtype=np.float16)
            keep.append(a)
            setattr(w, f, a.ctypes.data)
        self._w_keep = keep
        ret = _load().sk_gemma4_load_weights(self._h, ctypes.byref(w))
        if ret:
            raise RuntimeError(f"sk_gemma4_load_weights failed: {ret}")

    def reset(self):
        _load().sk_gemma4_reset(self._h)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """input_ids: (batch*seq,) or (batch, seq) int32. Returns (batch,) int32 argmax."""
        ids = np.ascontiguousarray(input_ids, dtype=np.int32).reshape(-1)
        seq = ids.size // self.cfg.batch
        out = np.empty((self.cfg.batch,), dtype=np.int32)
        ret = _load().sk_gemma4_forward(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            seq,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if ret:
            raise RuntimeError(f"sk_gemma4_forward failed: {ret}")
        return out

    def generate(self, input_ids: np.ndarray, max_new_tokens: int) -> list:
        """Greedy generation: prefill once, then decode-step. Single batch."""
        self.reset()
        toks = list(self.forward(input_ids))  # prefill returns argmax of last
        last = int(toks[0])
        out = [last]
        for _ in range(max_new_tokens - 1):
            nxt = self.forward(np.array([last], dtype=np.int32))
            last = int(nxt[0])
            out.append(last)
        return out

    def close(self):
        if self._h:
            _load().sk_gemma4_destroy(self._h)
            self._h = None
            self._w_keep = None

    def __del__(self):
        try: self.close()
        except Exception: pass
