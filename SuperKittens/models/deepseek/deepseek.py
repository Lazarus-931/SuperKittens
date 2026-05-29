"""DeepSeek V3 / V4 Flash inference handle."""
from __future__ import annotations
import ctypes, os
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from SuperKittens.inference.c_binder import bind, optional, CtypesConfig
from SuperKittens.inference.generation import Model


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
        # V3 additions — order MUST match sk_deepseek_config in launcher.h.
        ("has_q_lora",             ctypes.c_uint32),
        ("router_has_bias",        ctypes.c_uint32),
        ("rope_interleave",        ctypes.c_uint32),
        ("norm_topk_prob",         ctypes.c_uint32),
        ("n_group",                ctypes.c_uint32),
        ("topk_group",             ctypes.c_uint32),
        ("routed_scaling_factor",  ctypes.c_float),
        ("mscale_all_dim",         ctypes.c_float),
        ("rope_scaling_factor",    ctypes.c_float),
        ("first_k_dense_replace",  ctypes.c_uint32),
        ("dense_n_int",            ctypes.c_uint32),
    ]


_WEIGHT_FIELDS = (
    "w_embed", "w_lm_head",
    "w_pre_attn_norm",
    "w_q_a", "w_q_a_norm", "w_q_b",
    "w_kv_a", "w_kv_a_norm", "w_kv_b",
    "w_o", "w_pre_mlp_norm", "w_final_norm",
    "w_shared_gate", "w_shared_up", "w_shared_down",
    "w_router", "router_bias",
    "w_gate", "w_up", "w_down",
)


class _Weights(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in _WEIGHT_FIELDS]


DEEPSEEK_ABI = {
    "create":       ([ctypes.POINTER(_Config)], ctypes.c_void_p),
    "load_weights": ([ctypes.c_void_p, ctypes.POINTER(_Weights)], ctypes.c_int),
    "load_gguf":    optional([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
    "forward":      ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                      ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)], ctypes.c_int),
    "reset":        ([ctypes.c_void_p], None),
    "destroy":      ([ctypes.c_void_p], None),
}


_lib = None
def _load():
    global _lib
    if _lib is None:
        _lib = bind("deepseek", DEEPSEEK_ABI)
    return _lib


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class Config(CtypesConfig):
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

    # V3 additions.
    has_q_lora: int            = 1
    router_has_bias: int       = 1
    rope_interleave: int       = 1
    norm_topk_prob: int        = 1
    n_group: int               = 8
    topk_group: int            = 4
    routed_scaling_factor: float = 2.5
    mscale_all_dim: float      = 1.0
    rope_scaling_factor: float = 40.0
    first_k_dense_replace: int = 3
    dense_n_int: int           = 10944  # leading-dense-layer MLP width (V2-Lite)

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

    @property
    def dk(self) -> int: return self.qk_nope_dim + self.qk_rope_dim


# ─── Main handle ────────────────────────────────────────────────────────────

class DeepSeek(Model):
    """Stateful DeepSeek V4 Flash inference handle."""

    _repr_fields = (("L", "n_layers"), ("D", "d_model"), ("H", "n_heads"),
                    ("E", "n_expert"), ("top_k", "top_k"))

    def __init__(self, config: Config | str | None = None):
        if config is None:
            self.cfg = Config()
        elif isinstance(config, str):
            self.cfg = Config.preset(config)
        else:
            self.cfg = config
        lib = _load()
        self._destroy_fn = lib.sk_deepseek_destroy
        self._cstruct = self.cfg.to_c(_Config)
        self._h = lib.sk_deepseek_create(ctypes.byref(self._cstruct))
        if not self._h:
            raise RuntimeError("sk_deepseek_create failed (likely missing PSO or "
                               "unsupported flash_attn dk/dv shape)")
        self._w_keep: list[np.ndarray] | None = None
        self._last_token: int | None = None
        self._tok = None
        self.tokenizer = None
        self.vocab_size = self.cfg.vocab_size

    # ─── factory shortcuts ───
    @classmethod
    def test_config(cls) -> "DeepSeek":
        return cls(Config.preset("test"))

    # ─── weight loading ───
    def load_weights(self, weights: dict[str, np.ndarray] | None = None,
                     **kw_weights: np.ndarray) -> None:
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

    def load_gguf(self, path: str) -> None:
        """Load V2-Lite weights from a GGUF (dense dequant→fp16, routed experts
        kept quantized: gate/up Q4_K, down Q8_0)."""
        lib = _load()
        if not hasattr(lib, "sk_deepseek_load_gguf"):
            raise RuntimeError("libsk.dylib has no sk_deepseek_load_gguf; rebuild dylib")
        rc = lib.sk_deepseek_load_gguf(self._h, str(path).encode())
        if rc:
            raise RuntimeError(f"sk_deepseek_load_gguf failed: {rc}")

    # ─── state ───
    def reset(self) -> None:
        _load().sk_deepseek_reset(self._h)
        self._last_token = None

    @property
    def required_weights(self) -> tuple[str, ...]:
        return _WEIGHT_FIELDS

    # ─── inference ───
    def _forward(self, input_ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int32)).reshape(-1)
        seq = ids.size // self.cfg.batch
        out = np.empty((self.cfg.batch,), dtype=np.int32)
        rc = _load().sk_deepseek_forward(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            seq,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc: raise RuntimeError(f"forward failed: {rc}")
        self._last_token = int(out[0])
        return out

    def forward(self, input_ids) -> int:
        return int(self._forward(input_ids)[0])

    # ── tokenizer + chat API ─────────────────────────────────────────
    def attach_tokenizer(self, tokenizer) -> "DeepSeek":
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

    # ── factory: build from a central-registry ModelSpec ─────────────
    @classmethod
    def from_spec(cls, spec, **overrides) -> "DeepSeek":
        """Build a DeepSeek-V2-Lite model from a central-registry ModelSpec and a
        Q4_K_M GGUF. Routed experts stay quantized (gate/up Q4_K, down Q8_0);
        dense weights dequant to fp16 in the C loader. Attaches the tokenizer.

        Pass ``gguf=<path>`` to point at the GGUF; otherwise the snapshot dir
        (or ~/qwen-gguf) is scanned for ``DeepSeek-V2-Lite*Q4_K*.gguf``.
        """
        import json
        from dataclasses import fields
        sk_root = Path(__file__).resolve().parents[3]
        gguf_override = overrides.pop("gguf", None)
        snap = Path(overrides.pop("snapshot", None)
                    or (sk_root / "SuperKittens" / "model_weights" / spec.weight_dir))

        d = dict(spec.dims)
        cfg_json = snap / "config.json"
        if cfg_json.exists():
            j = json.loads(cfg_json.read_text())
            d.update(
                n_layers      = j.get("num_hidden_layers", d.get("n_layers")),
                d_model       = j.get("hidden_size",       d.get("d_model")),
                n_heads       = j.get("num_attention_heads", d.get("n_heads")),
                qk_nope_dim   = j.get("qk_nope_head_dim",  d.get("qk_nope_dim")),
                qk_rope_dim   = j.get("qk_rope_head_dim",  d.get("qk_rope_dim")),
                v_head_dim    = j.get("v_head_dim",        d.get("v_head_dim")),
                q_lora_rank   = (j.get("q_lora_rank") or 0),
                kv_lora_rank  = j.get("kv_lora_rank",      d.get("kv_lora_rank")),
                n_int         = j.get("moe_intermediate_size", d.get("n_int")),
                shared_n_int  = (j.get("n_shared_experts", 0)
                                 * j.get("moe_intermediate_size", 0)
                                 or d.get("shared_n_int")),
                dense_n_int   = j.get("intermediate_size", d.get("dense_n_int", 10944)),
                n_expert      = j.get("n_routed_experts",  d.get("n_expert")),
                top_k         = j.get("num_experts_per_tok", d.get("top_k")),
                vocab_size    = j.get("vocab_size",        d.get("vocab_size")),
                eps           = j.get("rms_norm_eps",      d.get("eps", 1e-6)),
                first_k_dense_replace = j.get("first_k_dense_replace",
                                              d.get("first_k_dense_replace", 1)),
            )
            rs = j.get("rope_scaling") or {}
            if rs:
                d["rope_scaling_factor"] = float(rs.get("factor", d.get("rope_scaling_factor", 1.0)))
                d["mscale_all_dim"]      = float(rs.get("mscale_all_dim",
                                                        d.get("mscale_all_dim", 1.0)))
                d["rope_n_ctx_orig"]     = int(rs.get("original_max_position_embeddings",
                                                      d.get("rope_n_ctx_orig", 4096)))
            d["rope_freq_base"] = float(j.get("rope_theta", d.get("rope_freq_base", 10000.0)))
            d["has_q_lora"]      = 1 if (j.get("q_lora_rank") or 0) > 0 else 0
            d["norm_topk_prob"]  = int(bool(j.get("norm_topk_prob", False)))
            d["routed_scaling_factor"] = float(j.get("routed_scaling_factor", 1.0))
            d["router_has_bias"] = 0  # V2-Lite: no e_score_correction_bias
            d["rope_interleave"] = 1
            d["n_group"]         = int(j.get("n_group") or 0)
            d["topk_group"]      = int(j.get("topk_group") or 0)

        # Routed experts are Q4_K(gate/up) + Q8_0(down) blocks → moe_quant=2.
        d["moe_quant"] = 2
        # Keep the KV cache modest on a 16 GB box (decode-only run).
        d.setdefault("seq_max", 512)
        d.setdefault("cache_max", 512)
        for k in ("seq_max", "cache_max"):
            if k in overrides:
                d[k] = overrides.pop(k)

        allowed = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in d.items() if k in allowed and v is not None})
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        # Resolve the GGUF path.
        gguf_path = None
        if gguf_override:
            gguf_path = Path(gguf_override)
        else:
            search = [snap, sk_root / "SuperKittens" / "model_weights",
                      Path.home() / "qwen-gguf"]
            for base in search:
                if not base.exists():
                    continue
                cands = sorted(base.glob("DeepSeek-V2-Lite*Q4_K*.gguf"))
                if cands:
                    gguf_path = cands[0]; break
        if gguf_path is None or not Path(gguf_path).exists():
            raise FileNotFoundError(
                "DeepSeek-V2-Lite Q4_K_M GGUF not found. Pass gguf=<path> or place "
                "DeepSeek-V2-Lite*Q4_K*.gguf under the snapshot dir or ~/qwen-gguf.")

        m = cls(cfg)
        m.load_gguf(str(gguf_path))

        if spec.tokenizer_family:
            try:
                from SuperKittens.models.load.tokenizer import Tokenizer
                json_path = snap / "tokenizer.json"
                sp_path   = snap / "tokenizer.model"
                if json_path.exists():
                    m._tok = Tokenizer.from_hf_json(str(json_path), family=spec.tokenizer_family)
                elif sp_path.exists():
                    m._tok = Tokenizer.from_sentencepiece(str(sp_path), family=spec.tokenizer_family)
                else:
                    print(f"[deepseek] no tokenizer.json/.model in {snap}")
            except Exception as e:
                print(f"[deepseek] tokenizer attach failed: {e}")
        m.tokenizer = m._tok
        return m

    def chat(self, text: str | list, *, max_new_tokens: int = 64) -> str:
        if self._tok is None:
            raise RuntimeError("attach_tokenizer() first")
        msgs = [{"role": "user", "content": text}] if isinstance(text, str) else text
        ids = self._tok.encode_chat(msgs)
        out_ids = self.generate(ids, max_new_tokens=max_new_tokens,
                                eos_id=getattr(self._tok, "eos_id", None))
        # Strip the prompt prefix; only return the newly generated portion.
        return self._tok.decode(out_ids[len(ids):] if len(out_ids) > len(ids) else out_ids)

