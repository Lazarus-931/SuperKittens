"""gemma4.py — ctypes wrapper around the Gemma 4 launcher (libsk.dylib)."""
from __future__ import annotations
import ctypes, os, json, warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from SuperKittens.inference.c_binder import bind, optional


_VARIANT_TO_DIR = {
    "e2b": "gemma-4-E2B-it",
    "e4b": "gemma-4-E4B-it",
    "26b": "gemma-4-26B-it",
    "31b": "gemma-4-31B-it",
}


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
        ("ple_dim",            ctypes.c_uint32),
        ("has_ple",            ctypes.c_int),
        ("eps",                ctypes.c_float),
        ("final_logit_softcap", ctypes.c_float),
        ("use_double_wide_mlp",  ctypes.c_uint32),
        ("num_kv_shared_layers", ctypes.c_uint32),
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


GEMMA4_ABI = {
    "create":                 ([ctypes.POINTER(_Config)], ctypes.c_void_p),
    "load_weights":           ([ctypes.c_void_p, ctypes.POINTER(_Weights)], ctypes.c_int),
    "forward":                ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)], ctypes.c_int),
    "reset":                  ([ctypes.c_void_p], None),
    "destroy":                ([ctypes.c_void_p], None),
    "load_safetensors":       ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
    "load_safetensors_index": ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
    "set_rope_tables":        ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
    "get_last_logits":        ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
    "set_dump_enabled":       ([ctypes.c_void_p, ctypes.c_int], None),
    "dump_layer":             ([ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p], ctypes.c_int),
}


_lib = None
def _load():
    global _lib
    if _lib is None:
        _lib = bind("gemma4", GEMMA4_ABI)
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
    head_dim_global: int = 256
    window: int = 4096
    prope_p_pairs: int = 64
    vocab_size: int = 262144
    has_ple: bool = False
    ple_dim: int = 256
    eps: float = 1e-6
    rope_theta_global: float = 1_000_000.0
    rope_theta_local: float = 10_000.0
    final_logit_softcap: float = 0.0
    partial_rotary_factor_full: float = 0.25
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    layer_types: tuple = ()
    batch: int = 1
    seq_max: int = 256       # prefill cap; sized for scratch buffers (per-dispatch overhead scales with bound buffer size on Apple GPUs)
    cache_max: int = 8192


def _bake_rope(cache_max: int, head_dim: int, theta: float):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(cache_max, dtype=np.float64)
    angles = np.outer(pos, inv_freq)
    # Return bf16 (uint16 with top half of fp32 word).
    def to_bf16(arr_f32):
        u32 = arr_f32.astype(np.float32).view(np.uint32)
        # Round-to-nearest-even: bias by 0x7fff + (LSB of result).
        rb = 0x00007fff + ((u32 >> 16) & 1)
        out = ((u32 + rb) >> 16).astype(np.uint16)
        return out
    c = to_bf16(np.cos(angles))
    s = to_bf16(np.sin(angles))
    return c, s


def _preset(name: str) -> Gemma4Config:
    n = name.lower()
    if n in ("e2b",):
        return Gemma4Config(n_layers=35, local_period=5, d_model=1536, n_int=6144,
                            n_heads=8, n_kv_heads_local=1, n_kv_heads_global=1,
                            head_dim_local=256, head_dim_global=512,
                            vocab_size=262144, window=4096, has_ple=True, ple_dim=256)
    if n in ("e4b",):
        return Gemma4Config(n_layers=35, local_period=6, d_model=2560, n_int=10240,
                            n_heads=8, n_kv_heads_local=2, n_kv_heads_global=2,
                            head_dim_local=256, head_dim_global=256,
                            vocab_size=262144, window=4096, has_ple=True, ple_dim=256)
    # TODO: verify against real config
    if n in ("26b", "26b-a4b"):
        return Gemma4Config(n_layers=60, local_period=6, d_model=4608, n_int=12288,
                            n_heads=16, n_kv_heads_local=16, n_kv_heads_global=4,
                            window=1024, has_ple=False)
    # TODO: verify against real config
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
    cs.ple_dim            = c.ple_dim
    cs.has_ple            = 1 if c.has_ple else 0
    cs.eps                = c.eps
    cs.final_logit_softcap = float(c.final_logit_softcap)
    cs.use_double_wide_mlp  = 1 if c.use_double_wide_mlp else 0
    cs.num_kv_shared_layers = int(c.num_kv_shared_layers)
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
        self._w_keep = None
        self._rope_keep = None
        self.tokenizer = None

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

    def set_dump_enabled(self, enabled: bool):
        _load().sk_gemma4_set_dump_enabled(self._h, 1 if enabled else 0)

    def dump(self, name: str) -> np.ndarray:
        """Pull a per-layer/post-norm/logits activation from the SK dump stash
        for the most recent forward(). Requires set_dump_enabled(True)."""
        if name == "logits":
            out = np.empty((self.cfg.vocab_size,), dtype=np.uint16)
        elif name in ("L0.q_normed", "L0.q_rope", "L0.attn_pre"):
            out = np.empty((self.cfg.n_heads * self.cfg.head_dim_local,), dtype=np.uint16)
        elif name in ("L0.k_normed", "L0.k_rope"):
            out = np.empty((self.cfg.n_kv_heads_local * self.cfg.head_dim_local,), dtype=np.uint16)
        elif name == "L1.qkv_pre_norm":
            qkvN = (self.cfg.n_heads + 2 * self.cfg.n_kv_heads_local) * self.cfg.head_dim_local
            out = np.empty((qkvN,), dtype=np.uint16)
        elif name in ("L0.ple_gate_out", "L0.ple_gated", "L0.per_layer_inputs"):
            out = np.empty((self.cfg.ple_dim,), dtype=np.uint16)
        elif name == "L0.ple_proj_back":
            out = np.empty((2 * self.cfg.d_model,), dtype=np.uint16)
        else:
            out = np.empty((self.cfg.d_model,), dtype=np.uint16)
        rc = _load().sk_gemma4_dump_layer(self._h, name.encode(), out.ctypes.data)
        if rc:
            raise RuntimeError(f"sk_gemma4_dump_layer({name!r}) failed: {rc}")
        if name == "L0.ple_proj_back":
            return out.view(np.float32).astype(np.float32)
        # Interpret as bf16 -> fp32 -> fp16 for downstream consumers.
        u32 = out.astype(np.uint32) << 16
        f32 = u32.view(np.float32)
        return f32.astype(np.float16)

    def last_logits(self) -> np.ndarray:
        out = np.empty((self.cfg.vocab_size,), dtype=np.uint16)
        rc = _load().sk_gemma4_get_last_logits(self._h, out.ctypes.data)
        if rc:
            raise RuntimeError(f"sk_gemma4_get_last_logits failed: {rc}")
        u32 = out.astype(np.uint32) << 16
        return u32.view(np.float32).astype(np.float16)

    def _sample(self, logits: np.ndarray, temperature: float, top_p: float,
                top_k: int | None, rng: np.random.Generator) -> int:
        if temperature <= 0.0:
            return int(np.argmax(logits))
        x = logits.astype(np.float32) / temperature
        x -= x.max()
        p = np.exp(x); p /= p.sum()
        if top_k is not None and top_k > 0 and top_k < p.size:
            idx = np.argpartition(p, -top_k)[-top_k:]
            mask = np.zeros_like(p); mask[idx] = p[idx]
            p = mask / mask.sum()
        if 0.0 < top_p < 1.0:
            order = np.argsort(p)[::-1]
            ps = p[order]
            cum = np.cumsum(ps)
            cutoff = np.searchsorted(cum, top_p) + 1
            keep = order[:cutoff]
            mask = np.zeros_like(p); mask[keep] = p[keep]
            p = mask / mask.sum()
        return int(rng.choice(p.size, p=p))

    def generate(self, input_ids, *, max_new_tokens: int = 64,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int | None = None, eos_id: int | None = None,
                 seed: int = 0) -> list:
        rng = np.random.default_rng(seed)
        ids = np.asarray(input_ids, dtype=np.int32).reshape(-1)
        self.reset()
        argmax_first = self.forward(ids)
        if temperature <= 0.0 and (top_p >= 1.0 or top_p <= 0.0) and not top_k:
            first = int(argmax_first[0])
        else:
            first = self._sample(self.last_logits(), temperature, top_p, top_k, rng)
        out = [first]
        if eos_id is not None and first == eos_id:
            return out
        last = first
        for _ in range(max_new_tokens - 1):
            arg = self.forward(np.array([last], dtype=np.int32))
            if temperature <= 0.0 and (top_p >= 1.0 or top_p <= 0.0) and not top_k:
                last = int(arg[0])
            else:
                last = self._sample(self.last_logits(), temperature, top_p, top_k, rng)
            out.append(last)
            if eos_id is not None and last == eos_id:
                break
        return out

    def chat(self, prompt: str, **gen_kwargs) -> str:
        if not getattr(self, "tokenizer", None):
            raise RuntimeError("no tokenizer attached. use from_pretrained or set .tokenizer")
        ids = self.tokenizer.encode(prompt)
        eos = gen_kwargs.pop("eos_id", getattr(self.tokenizer, "eos_id", None))
        out_ids = self.generate(np.array(ids, dtype=np.int32), eos_id=eos, **gen_kwargs)
        return self.tokenizer.decode(out_ids)

    def attach_ple_table(self, ple_table: np.ndarray):
        """Hold reference to the (vocab_size, ple_dim) PLE lookup table for per-token gather."""
        self._ple_table = np.ascontiguousarray(ple_table, dtype=np.float16)

    def set_ple_table_for_input(self, input_ids):
        """Gather per-token PLE rows from the attached PLE table and upload via C entry.

        If the C entry isn't compiled in yet, warn and return without failing.
        """
        if not getattr(self.cfg, "has_ple", False):
            return
        tbl = getattr(self, "_ple_table", None)
        if tbl is None:
            warnings.warn("[gemma4] set_ple_table_for_input: no PLE table attached; call attach_ple_table() first")
            return
        ids = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int64).reshape(-1))
        gathered = np.ascontiguousarray(tbl[ids], dtype=np.float16)
        self._ple_input_keep = gathered
        lib = _load()
        fn = getattr(lib, "sk_gemma4_set_ple_input", None)
        if fn is None:
            warnings.warn("[gemma4] sk_gemma4_set_ple_input not present in libsk; PLE per-token upload skipped (orchestrator agent owns it)")
            return
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        fn.restype = ctypes.c_int
        rc = fn(self._h, gathered.ctypes.data, ctypes.c_uint32(ids.size))
        if rc:
            raise RuntimeError(f"sk_gemma4_set_ple_input failed: {rc}")

    def set_rope_tables(self, cos_local, sin_local, cos_global, sin_global):
        # Now uint16 bf16 bytes from _bake_rope.
        cl = np.ascontiguousarray(cos_local, dtype=np.uint16)
        sl = np.ascontiguousarray(sin_local, dtype=np.uint16)
        cg = np.ascontiguousarray(cos_global, dtype=np.uint16)
        sg = np.ascontiguousarray(sin_global, dtype=np.uint16)
        self._rope_keep = (cl, sl, cg, sg)
        rc = _load().sk_gemma4_set_rope_tables(
            self._h, cl.ctypes.data, sl.ctypes.data, cg.ctypes.data, sg.ctypes.data)
        if rc:
            raise RuntimeError(f"sk_gemma4_set_rope_tables failed: {rc}")

    @classmethod
    def from_pretrained(cls, variant: str = "e2b", name: str | None = None, **cfg_overrides):
        if name is None:
            name = variant
        if name in _VARIANT_TO_DIR:
            dir_name = _VARIANT_TO_DIR[name]
            variant = name
        else:
            dir_name = name
            variant = next((v for v, d in _VARIANT_TO_DIR.items() if d == name), "e4b")

        root = Path(__file__).resolve().parents[4] / "SuperKittens" / "model_weights" / dir_name
        if not root.exists():
            raise FileNotFoundError(
                f"{root} not found. Run: ./SuperKittens/models/gemma/gemma4/download.sh {variant}")

        cfg_path = root / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing {cfg_path}")
        hf = json.loads(cfg_path.read_text())
        text_cfg = hf.get("text_config", {}) if isinstance(hf.get("text_config"), dict) else {}
        def _get(key, default):
            if key in text_cfg: return text_cfg[key]
            if key in hf: return hf[key]
            return default
        cfg = _preset(variant)
        cfg.n_layers          = int(_get("num_hidden_layers", cfg.n_layers))
        cfg.d_model           = int(_get("hidden_size", cfg.d_model))
        cfg.n_int             = int(_get("intermediate_size", cfg.n_int))
        cfg.n_heads           = int(_get("num_attention_heads", cfg.n_heads))
        nkv                   = int(_get("num_key_value_heads", cfg.n_kv_heads_local))
        cfg.n_kv_heads_local  = nkv
        cfg.n_kv_heads_global = nkv
        cfg.head_dim_local    = int(_get("head_dim", cfg.head_dim_local))
        cfg.head_dim_global   = int(_get("global_head_dim", cfg.head_dim_global))
        cfg.ple_dim           = int(_get("hidden_size_per_layer_input", cfg.ple_dim))
        cfg.window            = int(_get("sliding_window", cfg.window))
        cfg.vocab_size        = int(_get("vocab_size", cfg.vocab_size))
        cfg.eps               = float(_get("rms_norm_eps", cfg.eps))
        cfg.rope_theta_global = float(_get("rope_theta", cfg.rope_theta_global))
        cfg.rope_theta_local  = float(_get("rope_local_base_freq",
                                _get("rope_local_theta",
                                _get("rope_theta_local", cfg.rope_theta_local))))
        cfg.final_logit_softcap = float(_get("final_logit_softcapping", 0.0) or 0.0)
        cfg.num_kv_shared_layers = int(_get("num_kv_shared_layers", 0))
        cfg.use_double_wide_mlp  = bool(_get("use_double_wide_mlp", False))
        lt = _get("layer_types", None)
        if isinstance(lt, list) and lt:
            cfg.layer_types = tuple(lt)
        rp = _get("rope_parameters", {})
        if isinstance(rp, dict):
            full = rp.get("full_attention", {})
            if isinstance(full, dict) and "partial_rotary_factor" in full:
                cfg.partial_rotary_factor_full = float(full["partial_rotary_factor"])
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

        m = cls(cfg)

        idx = root / "model.safetensors.index.json"
        single = root / "model.safetensors"
        if idx.exists():
            rc = _load().sk_gemma4_load_safetensors_index(m._h, str(idx).encode())
        elif single.exists():
            rc = _load().sk_gemma4_load_safetensors(m._h, str(single).encode())
        else:
            raise FileNotFoundError(f"no safetensors in {root}")
        if rc: raise RuntimeError(f"load failed: {rc}")

        cos_l, sin_l = _bake_rope(cfg.cache_max, cfg.head_dim_local,  cfg.rope_theta_local)
        cos_g, sin_g = _bake_rope(cfg.cache_max, cfg.head_dim_global, cfg.rope_theta_global)
        m.set_rope_tables(cos_l, sin_l, cos_g, sin_g)

        m.tokenizer = None
        try:
            from SuperKittens.models.load.tokenizer import Tokenizer
        except Exception as e:
            print(f"[gemma4] tokenizer module unavailable: {e}")
            return m
        sp_path = root / "tokenizer.model"
        json_path = root / "tokenizer.json"
        if sp_path.exists():
            try:
                m.tokenizer = Tokenizer.from_sentencepiece(str(sp_path))
            except Exception as e:
                print(f"[gemma4] sentencepiece attach failed: {e}")
        if m.tokenizer is None and json_path.exists():
            try:
                m.tokenizer = Tokenizer.from_hf_json(str(json_path), family="gemma")
            except Exception as e:
                print(f"[gemma4] hf-json attach failed: {e}")
        return m

    def close(self):
        if self._h:
            _load().sk_gemma4_destroy(self._h)
            self._h = None
            self._w_keep = None

    def __del__(self):
        try: self.close()
        except Exception: pass
