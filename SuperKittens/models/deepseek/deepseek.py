"""deepseek.py — clean Python API for DeepSeek V4 Flash inference.

Usage:

    from models.deepseek.deepseek import DeepSeek

    # 1) Preset (most common)
    with DeepSeek("v4_flash") as m:
        m.load_weights(state_dict)             # dict mapping weight_name → np.ndarray
        tokens = m.generate([1, 2, 3], max_new_tokens=64)

    # 2) Tiny test config (no weights needed for API exercise)
    with DeepSeek.test_config() as m:
        m.load_random_weights(seed=0)
        token_id = m.forward([1, 2, 3])

    # 3) Explicit prefill/decode loop
    with DeepSeek("v4_flash") as m:
        m.load_weights(state_dict)
        m.prefill([1, 2, 3])                   # extends KV cache, returns last token
        for _ in range(64):
            next_tok = m.decode_step()
            print(next_tok)
"""
from __future__ import annotations
import ctypes, os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict


# ─── C ABI ──────────────────────────────────────────────────────────────────

class _Config(ctypes.Structure):
    _fields_ = [
        ("batch",              ctypes.c_uint32),
        ("seq_max",            ctypes.c_uint32),
        ("cache_max",          ctypes.c_uint32),
        ("n_layers",           ctypes.c_uint32),
        ("d_model",            ctypes.c_uint32),
        ("n_heads",            ctypes.c_uint32),
        ("qk_nope_dim",        ctypes.c_uint32),
        ("qk_rope_dim",        ctypes.c_uint32),
        ("v_head_dim",         ctypes.c_uint32),
        ("q_lora_rank",        ctypes.c_uint32),
        ("kv_lora_rank",       ctypes.c_uint32),
        ("n_int",              ctypes.c_uint32),
        ("shared_n_int",       ctypes.c_uint32),
        ("n_expert",           ctypes.c_uint32),
        ("top_k",              ctypes.c_uint32),
        ("vocab_size",         ctypes.c_uint32),
        ("moe_quant",          ctypes.c_uint32),    # 0 = fp16, 1 = INT2_DS4
        ("rope_n_ctx_orig",    ctypes.c_int32),
        ("rope_freq_base",     ctypes.c_float),
        ("rope_freq_scale",    ctypes.c_float),
        ("rope_ext_factor",    ctypes.c_float),
        ("rope_attn_factor",   ctypes.c_float),
        ("rope_beta_fast",     ctypes.c_float),
        ("rope_beta_slow",     ctypes.c_float),
        ("eps",                ctypes.c_float),
    ]


_WEIGHT_FIELDS = (
    "w_embed",
    "w_pre_attn_norm",
    "w_q_a", "w_q_a_norm", "w_q_b",
    "w_kv_a", "w_kv_a_norm", "w_kv_b",
    "w_o", "w_pre_mlp_norm", "w_final_norm",
    "w_shared_gate", "w_shared_up", "w_shared_down",
    "w_router", "w_gate", "w_up", "w_down",
)


class _Weights(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in _WEIGHT_FIELDS]


_lib = None
def _load():
    global _lib
    if _lib is not None: return _lib
    dylib = os.environ.get("SK_DYLIB",
                           str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"))
    _lib = ctypes.CDLL(dylib)
    _lib.sk_deepseek_create.argtypes        = [ctypes.POINTER(_Config)]
    _lib.sk_deepseek_create.restype         = ctypes.c_void_p
    _lib.sk_deepseek_load_weights.argtypes  = [ctypes.c_void_p, ctypes.POINTER(_Weights)]
    _lib.sk_deepseek_load_weights.restype   = ctypes.c_int
    _lib.sk_deepseek_forward.argtypes       = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_int32),
                                               ctypes.c_uint32,
                                               ctypes.POINTER(ctypes.c_int32)]
    _lib.sk_deepseek_forward.restype        = ctypes.c_int
    _lib.sk_deepseek_reset.argtypes         = [ctypes.c_void_p]
    _lib.sk_deepseek_reset.restype          = None
    _lib.sk_deepseek_destroy.argtypes       = [ctypes.c_void_p]
    _lib.sk_deepseek_destroy.restype        = None
    return _lib


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Model dimensions. Use `Config.preset(name)` for known variants."""
    n_layers: int       = 60
    d_model: int        = 7168
    n_heads: int        = 128
    qk_nope_dim: int    = 128
    qk_rope_dim: int    = 64
    v_head_dim: int     = 128
    q_lora_rank: int    = 1536
    kv_lora_rank: int   = 512
    n_int: int          = 2048
    shared_n_int: int   = 2048
    n_expert: int       = 256
    top_k: int          = 8
    vocab_size: int     = 129280

    rope_n_ctx_orig: int = 4096
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    rope_ext_factor: float = 0.0
    rope_attn_factor: float = 1.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0

    eps: float = 1e-6
    batch: int = 1
    seq_max: int = 8192
    cache_max: int = 8192
    # 0 = fp16 routed-expert weights; 1 = INT2_DS4 (IQ2_XXS up/gate, Q2_K down).
    moe_quant: int = 0

    @classmethod
    def preset(cls, name: str) -> "Config":
        n = name.lower().replace("-", "_")
        if n in ("v4_flash", "deepseek_v4_flash", "v4"):
            return cls()  # defaults are V4 Flash
        if n in ("test", "tiny"):
            # Smallest config that exercises every kernel path.
            return cls(n_layers=2, d_model=512, n_heads=4,
                       qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128,
                       q_lora_rank=128, kv_lora_rank=128,
                       n_int=256, shared_n_int=256, n_expert=8, top_k=2,
                       vocab_size=1024, seq_max=16, cache_max=64)
        raise ValueError(f"unknown DeepSeek preset: {name!r}")

    def _to_c(self) -> _Config:
        cs = _Config()
        for k, v in asdict(self).items(): setattr(cs, k, v)
        return cs

    @property
    def dk(self) -> int: return self.qk_nope_dim + self.qk_rope_dim


# ─── Main handle ────────────────────────────────────────────────────────────

class DeepSeek:
    """Stateful DeepSeek V4 Flash inference handle."""

    def __init__(self, config: Config | str | None = None):
        if config is None:
            self.cfg = Config()
        elif isinstance(config, str):
            self.cfg = Config.preset(config)
        else:
            self.cfg = config
        lib = _load()
        self._cstruct = self.cfg._to_c()
        self._h = lib.sk_deepseek_create(ctypes.byref(self._cstruct))
        if not self._h:
            raise RuntimeError("sk_deepseek_create failed (likely missing PSO or "
                               "unsupported flash_attn dk/dv shape)")
        self._w_keep: list[np.ndarray] | None = None
        self._last_token: int | None = None
        self._tok = None

    # ─── factory shortcuts ───
    @classmethod
    def test_config(cls) -> "DeepSeek":
        return cls(Config.preset("test"))

    # ─── weight loading ───
    def load_weights(self, weights: dict[str, np.ndarray] | None = None,
                     **kw_weights: np.ndarray) -> None:
        """Pass weights as a dict or kwargs. All 18 weights are required."""
        if weights is None: weights = {}
        weights = {**weights, **kw_weights}
        keep: list[np.ndarray] = []
        w = _Weights()
        for name in _WEIGHT_FIELDS:
            a = weights.get(name)
            if a is None:
                raise ValueError(f"missing weight: {name}. Required: {_WEIGHT_FIELDS}")
            a = np.ascontiguousarray(a, dtype=np.float16)
            keep.append(a)
            setattr(w, name, a.ctypes.data)
        self._w_keep = keep
        rc = _load().sk_deepseek_load_weights(self._h, ctypes.byref(w))
        if rc: raise RuntimeError(f"sk_deepseek_load_weights failed: {rc}")

    def load_random_weights(self, seed: int = 0, scale: float = 0.02) -> None:
        """Fill all weights with small random fp16 values. For API/perf testing."""
        c = self.cfg
        dk = c.dk
        rng = np.random.default_rng(seed)
        def W(*shape): return (rng.standard_normal(shape) * scale).astype(np.float16)
        def O(*shape): return np.ones(shape, dtype=np.float16)
        self.load_weights({
            "w_embed":         W(c.vocab_size, c.d_model),
            "w_pre_attn_norm": O(c.n_layers, c.d_model),
            "w_q_a":           W(c.n_layers, c.d_model, c.q_lora_rank),
            "w_q_a_norm":      O(c.n_layers, c.q_lora_rank),
            "w_q_b":           W(c.n_layers, c.q_lora_rank, c.n_heads * dk),
            "w_kv_a":          W(c.n_layers, c.d_model, c.kv_lora_rank + c.qk_rope_dim),
            "w_kv_a_norm":     O(c.n_layers, c.kv_lora_rank),
            "w_kv_b":          W(c.n_layers, c.kv_lora_rank,
                                 c.n_heads * (c.qk_nope_dim + c.v_head_dim)),
            "w_o":             W(c.n_layers, c.n_heads * c.v_head_dim, c.d_model),
            "w_pre_mlp_norm":  O(c.n_layers, c.d_model),
            "w_final_norm":    O(c.d_model),
            "w_shared_gate":   W(c.n_layers, c.d_model, c.shared_n_int),
            "w_shared_up":     W(c.n_layers, c.d_model, c.shared_n_int),
            "w_shared_down":   W(c.n_layers, c.shared_n_int, c.d_model),
            "w_router":        W(c.n_layers, c.d_model, c.n_expert),
            "w_gate":          W(c.n_layers, c.n_expert, c.n_int, c.d_model),
            "w_up":            W(c.n_layers, c.n_expert, c.n_int, c.d_model),
            "w_down":          W(c.n_layers, c.n_expert, c.d_model, c.n_int),
        })

    # ─── state ───
    def reset(self) -> None:
        _load().sk_deepseek_reset(self._h)
        self._last_token = None

    @property
    def required_weights(self) -> tuple[str, ...]:
        return _WEIGHT_FIELDS

    # ─── inference ───
    def forward(self, input_ids) -> int:
        """Forward pass over `seq` tokens; returns argmax token id."""
        ids = np.asarray(input_ids, dtype=np.int32).reshape(-1)
        seq = ids.size // self.cfg.batch
        out = np.empty((self.cfg.batch,), dtype=np.int32)
        rc = _load().sk_deepseek_forward(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            seq,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc: raise RuntimeError(f"forward failed: {rc}")
        self._last_token = int(out[0])
        return self._last_token

    def prefill(self, input_ids) -> int:
        """Eats the prompt and returns the argmax of the last position."""
        self.reset()
        return self.forward(input_ids)

    def decode_step(self) -> int:
        """One decode step continuing from the last forward's argmax."""
        if self._last_token is None:
            raise RuntimeError("decode_step called before prefill/forward")
        return self.forward([self._last_token])

    def generate(self, input_ids, max_new_tokens: int = 64,
                 *, stop_on_eos: bool = True) -> list[int]:
        """Greedy decode: prefill, then loop. Stops at max_new_tokens or EOS
        (when a tokenizer with an EOS id is attached and stop_on_eos=True).

        Sampling is currently argmax only; temperature / top-p / top-k will
        plug in here once the GPU sampler kernel is wired."""
        out: list[int] = [self.prefill(input_ids)]
        if stop_on_eos and self._tok and self._tok.is_eos(out[-1]):
            return out
        for _ in range(max_new_tokens - 1):
            tok = self.decode_step()
            out.append(tok)
            if stop_on_eos and self._tok and self._tok.is_eos(tok):
                break
        return out

    # ── tokenizer + chat API ─────────────────────────────────────────
    def attach_tokenizer(self, tokenizer) -> "DeepSeek":
        """Attach a tokenizer (DeepSeekTokenizer or any object with
        encode/decode/is_eos methods). Enables tokenize/detokenize/chat."""
        self._tok = tokenizer
        return self

    def tokenize(self, text: str, *, bos: bool = True) -> list[int]:
        if self._tok is None:
            raise RuntimeError("No tokenizer attached. Call attach_tokenizer() first.")
        return self._tok.encode(text, bos=bos)

    def detokenize(self, ids) -> str:
        if self._tok is None:
            raise RuntimeError("No tokenizer attached. Call attach_tokenizer() first.")
        return self._tok.decode(list(ids))

    def chat(self, text: str | list, *, max_new_tokens: int = 64) -> str:
        """One-shot chat. `text` is either a user-message string or a list of
        `{role, content}` dicts. Returns the assistant's reply as text.

        Greedy decoding. Use generate() directly if you want full control."""
        if self._tok is None:
            raise RuntimeError("attach_tokenizer() first")
        msgs = [{"role": "user", "content": text}] if isinstance(text, str) else text
        ids = self._tok.encode_chat(msgs)
        out_ids = self.generate(ids, max_new_tokens=max_new_tokens)
        # Strip the prompt prefix; only return the newly generated portion.
        return self._tok.decode(out_ids[len(ids):] if len(out_ids) > len(ids) else out_ids)

    # ─── lifecycle ───
    def close(self) -> None:
        if self._h:
            _load().sk_deepseek_destroy(self._h)
            self._h = None
            self._w_keep = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def __del__(self):
        try: self.close()
        except Exception: pass

    def __repr__(self) -> str:
        c = self.cfg
        return (f"DeepSeek(L={c.n_layers}, D={c.d_model}, H={c.n_heads}, "
                f"dk={c.dk}, dv={c.v_head_dim}, E={c.n_expert}, top_k={c.top_k})")
