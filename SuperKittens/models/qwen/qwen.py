"""qwen.py — clean Python API for Qwen3-32B (dense) inference.

Usage:
    from models.qwen.qwen import Qwen
    with Qwen("32b") as m:
        m.load_random_weights(seed=0)
        toks = m.generate([1,2,3], max_new_tokens=8)
"""
from __future__ import annotations
import ctypes, os
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from SuperKittens.inference.generation import Model
from SuperKittens.inference.c_binder import bind, optional, CtypesConfig


class _Config(ctypes.Structure):
    _fields_ = [
        ("batch",              ctypes.c_uint32),
        ("seq_max",            ctypes.c_uint32),
        ("cache_max",          ctypes.c_uint32),
        ("n_layers",           ctypes.c_uint32),
        ("d_model",            ctypes.c_uint32),
        ("n_heads",            ctypes.c_uint32),
        ("n_kv_heads",         ctypes.c_uint32),
        ("head_dim",           ctypes.c_uint32),
        ("n_int",              ctypes.c_uint32),
        ("vocab_size",         ctypes.c_uint32),
        ("rope_n_ctx_orig",    ctypes.c_int32),
        ("rope_freq_base",     ctypes.c_float),
        ("rope_freq_scale",    ctypes.c_float),
        ("rope_ext_factor",    ctypes.c_float),
        ("rope_attn_factor",   ctypes.c_float),
        ("rope_beta_fast",     ctypes.c_float),
        ("rope_beta_slow",     ctypes.c_float),
        ("eps",                ctypes.c_float),
        ("tie_word_embeddings", ctypes.c_uint32),
    ]


_WEIGHT_FIELDS = (
    "w_embed",
    "w_pre_attn_norm",
    "w_qkv",
    "w_q_norm",         # Qwen3-specific per-head Q-norm γ
    "w_k_norm",         # Qwen3-specific per-head K-norm γ
    "w_o",
    "w_pre_mlp_norm",
    "w_final_norm",
    "w_gate",
    "w_up",
    "w_down",
    "w_lm_head",
)


class _Weights(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in _WEIGHT_FIELDS]


QWEN_ABI = {
    "create":           ([ctypes.POINTER(_Config)], ctypes.c_void_p),
    "load_weights":     ([ctypes.c_void_p, ctypes.POINTER(_Weights)], ctypes.c_int),
    "forward":          ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                          ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)], ctypes.c_int),
    "reset":            ([ctypes.c_void_p], None),
    "destroy":          ([ctypes.c_void_p], None),
    "load_safetensors":       ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
    "load_safetensors_index": optional([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
    "load_gguf":              optional([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
    "set_rope_tables":  optional([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
    "get_last_logits":  optional([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
}


_lib = None
def _load():
    global _lib
    if _lib is None:
        _lib = bind("qwen", QWEN_ABI)
    return _lib


@dataclass
class Config(CtypesConfig):
    # Qwen3-32B dense defaults (per HF config.json).
    n_layers: int       = 64
    d_model: int        = 5120
    n_heads: int        = 64
    n_kv_heads: int     = 8
    head_dim: int       = 128
    n_int: int          = 27392
    vocab_size: int     = 151936

    rope_n_ctx_orig: int = 32768
    rope_freq_base: float = 1_000_000.0
    rope_freq_scale: float = 1.0
    rope_ext_factor: float = 0.0
    rope_attn_factor: float = 1.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0

    eps: float = 1e-6
    batch: int = 1
    seq_max: int = 8192
    cache_max: int = 32768
    tie_word_embeddings: int = 1  # Qwen3-0.6B and 32B share tied embeddings; set 0 for untied variants

    @classmethod
    def preset(cls, name: str) -> "Config":
        n = name.lower().replace("-", "_")
        if n in ("32b", "qwen3_32b", "v3_32b"):
            return cls()
        if n in ("test", "tiny"):
            # Smallest config that exercises every dispatch step.
            return cls(n_layers=2, d_model=512, n_heads=4, n_kv_heads=2,
                       head_dim=128, n_int=512, vocab_size=1024,
                       seq_max=16, cache_max=64,
                       rope_freq_base=1_000_000.0, rope_n_ctx_orig=64)
        raise ValueError(f"unknown Qwen preset: {name!r}")


class Qwen(Model):
    """Stateful Qwen3 (dense) inference handle."""

    _repr_fields = (("L", "n_layers"), ("D", "d_model"), ("H", "n_heads"),
                    ("hd", "head_dim"), ("n_int", "n_int"))

    def __init__(self, config: Config | str | None = None):
        if config is None: self.cfg = Config()
        elif isinstance(config, str): self.cfg = Config.preset(config)
        else: self.cfg = config
        lib = _load()
        self._destroy_fn = lib.sk_qwen_destroy
        self._cstruct = self.cfg.to_c(_Config)
        self._h = lib.sk_qwen_create(ctypes.byref(self._cstruct))
        if not self._h:
            raise RuntimeError("sk_qwen_create failed (missing PSO or wrong dims)")
        self._w_keep: list[np.ndarray] | None = None
        self._last_token: int | None = None
        self._tok = None
        self._rope_keep = None
        self.tokenizer = None
        self.vocab_size = self.cfg.vocab_size

    @classmethod
    def test_config(cls) -> "Qwen":
        return cls(Config.preset("test"))

    def load_weights(self, weights: dict[str, np.ndarray] | None = None,
                     **kw_weights: np.ndarray) -> None:
        if weights is None: weights = {}
        weights = {**weights, **kw_weights}
        keep: list[np.ndarray] = []
        w = _Weights()
        for name in _WEIGHT_FIELDS:
            a = weights.get(name)
            if a is None:
                if name == "w_lm_head":
                    # optional: only required when tie_word_embeddings == 0
                    setattr(w, name, 0)
                    continue
                raise ValueError(f"missing weight: {name}. Required: {_WEIGHT_FIELDS}")
            a = np.ascontiguousarray(a, dtype=np.float16)
            keep.append(a)
            setattr(w, name, a.ctypes.data)
        self._w_keep = keep
        rc = _load().sk_qwen_load_weights(self._h, ctypes.byref(w))
        if rc: raise RuntimeError(f"sk_qwen_load_weights failed: {rc}")

    def load_random_weights(self, seed: int = 0, scale: float = 0.02) -> None:
        c = self.cfg
        hd = c.head_dim
        qkv_N = (c.n_heads + 2 * c.n_kv_heads) * hd
        rng = np.random.default_rng(seed)
        def W(*shape): return (rng.standard_normal(shape) * scale).astype(np.float16)
        def O(*shape): return np.ones(shape, dtype=np.float16)
        self.load_weights({
            "w_embed":         W(c.vocab_size, c.d_model),
            "w_pre_attn_norm": O(c.n_layers, c.d_model),
            "w_qkv":           W(c.n_layers, c.d_model, qkv_N),
            "w_q_norm":        O(c.n_layers, hd),
            "w_k_norm":        O(c.n_layers, hd),
            "w_o":             W(c.n_layers, c.n_heads * hd, c.d_model),
            "w_pre_mlp_norm":  O(c.n_layers, c.d_model),
            "w_final_norm":    O(c.d_model),
            "w_gate":          W(c.n_layers, c.d_model, c.n_int),
            "w_up":            W(c.n_layers, c.d_model, c.n_int),
            "w_down":          W(c.n_layers, c.n_int, c.d_model),
        })

    def load_gguf(self, path: str) -> None:
        """Load Qwen3 weights from a GGUF file (e.g. q8_0). The file's projection
        dtype (Q8_0 / F16) is auto-detected; the launcher reallocates the affected
        weight buffers to the right size and routes M=1 GEMMs to q8_0_matvec."""
        lib = _load()
        if not hasattr(lib, "sk_qwen_load_gguf"):
            raise RuntimeError("libsk.dylib has no sk_qwen_load_gguf symbol; rebuild dylib")
        rc = lib.sk_qwen_load_gguf(self._h, str(path).encode())
        if rc:
            raise RuntimeError(f"sk_qwen_load_gguf failed: {rc}")

    def reset(self) -> None:
        _load().sk_qwen_reset(self._h)
        self._last_token = None

    def forward(self, input_ids) -> int:
        ids = np.asarray(input_ids, dtype=np.int32).reshape(-1)
        seq = ids.size // self.cfg.batch
        out = np.empty((self.cfg.batch,), dtype=np.int32)
        rc = _load().sk_qwen_forward(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            seq,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc: raise RuntimeError(f"forward failed: {rc}")
        self._last_token = int(out[0])
        return self._last_token

    def _forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Model-base contract: take int32 ids, return (batch,) int32 argmax."""
        ids = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int32)).reshape(-1)
        seq = ids.size // self.cfg.batch
        out = np.empty((self.cfg.batch,), dtype=np.int32)
        rc = _load().sk_qwen_forward(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            seq,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc: raise RuntimeError(f"forward failed: {rc}")
        self._last_token = int(out[0])
        return out

    def _last_logits(self) -> np.ndarray:
        out = np.empty((self.cfg.vocab_size,), dtype=np.float16)
        rc = _load().sk_qwen_get_last_logits(self._h, out.ctypes.data)
        if rc: raise RuntimeError(f"sk_qwen_get_last_logits failed: {rc}")
        return out

    def set_rope_tables(self, cos: np.ndarray, sin: np.ndarray) -> None:
        c = np.ascontiguousarray(cos, dtype=np.float16)
        s = np.ascontiguousarray(sin, dtype=np.float16)
        self._rope_keep = (c, s)
        rc = _load().sk_qwen_set_rope_tables(self._h, c.ctypes.data, s.ctypes.data)
        if rc: raise RuntimeError(f"sk_qwen_set_rope_tables failed: {rc}")

    def bake_and_set_rope(self) -> None:
        """Build RoPE cos/sin tables from cfg and upload to native handle."""
        half = self.cfg.head_dim // 2
        theta = self.cfg.rope_freq_base
        inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        pos = np.arange(self.cfg.cache_max, dtype=np.float64)
        angles = np.outer(pos, inv_freq)
        cos = np.cos(angles).astype(np.float16).copy()
        sin = np.sin(angles).astype(np.float16).copy()
        self.set_rope_tables(cos, sin)

    def chat(self, prompt, *, use_chat_template: bool = True, **gen_kwargs) -> str:
        """Run end-to-end chat. `prompt` may be str or list[{role,content}]."""
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer attached. use sk.load(...) or attach manually")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)
        if use_chat_template and hasattr(self.tokenizer, "chat"):
            try:
                ids = self.tokenizer.chat(messages, add_generation_prompt=True, bos=False)
            except Exception:
                ids = self.tokenizer.encode(messages[-1].get("content", ""))
        else:
            ids = self.tokenizer.encode(messages[-1].get("content", ""))
        eos = gen_kwargs.pop("eos_id", getattr(self.tokenizer, "eos_id", None))
        out_ids = self.generate(np.array(ids, dtype=np.int32), eos_id=eos, **gen_kwargs)
        return self.tokenizer.decode(out_ids)

