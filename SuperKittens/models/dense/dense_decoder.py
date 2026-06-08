"""dense_decoder.py — shared dense-transformer inference core.

This is the family-agnostic Python handle over the `sk_qwen_*` C-ABI launcher
(the in-tree dense-transformer core: pre-norm RMSNorm, packed-QKV GQA attention,
SwiGLU MLP, optionally per-head Q/K-norm). Each family ships a *thin* adapter
that subclasses :class:`DenseDecoder` and configures the core:

  * Qwen3 — per-head Q/K RMSNorm ON, plain RoPE.
  * Nemotron-Nano (Llama-arch) — Q/K-norm OFF, Llama-3.1 (llama3-scaled) RoPE,
    BOS-prefixed generation.

The acceptable division here is "separate adapter -> configures shared core":
the launcher/kernels are shared, the per-family identity (config flags, RoPE
bake, generation prelude) lives in the adapter. ``use_qk_norm`` is a generic
config field of the core, not a per-family conditional baked into one family's
forward.
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
        ("use_qk_norm",        ctypes.c_uint32),
        ("rope_interleaved",   ctypes.c_uint32),
    ]


_WEIGHT_FIELDS = (
    "w_embed",
    "w_pre_attn_norm",
    "w_qkv",
    "w_q_norm",         # per-head Q-norm γ (Qwen3; absent for Llama-arch)
    "w_k_norm",         # per-head K-norm γ (Qwen3; absent for Llama-arch)
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


# WHY: the shared dense launcher's C symbols are `sk_qwen_*` (the core was first
# landed for Qwen3). The adapter layer below is family-agnostic; only the C
# symbol prefix is named "qwen".
DENSE_ABI = {
    "create":           ([ctypes.POINTER(_Config)], ctypes.c_void_p),
    "load_weights":     ([ctypes.c_void_p, ctypes.POINTER(_Weights)], ctypes.c_int),
    "forward":          ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                          ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)], ctypes.c_int),
    "prefill_chunked":  optional([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                  ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.POINTER(ctypes.c_int32)], ctypes.c_int),
    "generate_n":       optional([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                  ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),
                                  ctypes.c_uint32, ctypes.c_int32], ctypes.c_int),
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
        _lib = bind("qwen", DENSE_ABI)
    return _lib


@dataclass
class Config(CtypesConfig):
    # Dense decoder dims (defaults are Qwen3-32B; adapters override via from_spec).
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
    tie_word_embeddings: int = 1  # 1 = tied LM head; 0 for untied variants
    use_qk_norm: int = 1          # 1 = per-head Q/K RMSNorm (Qwen3); 0 for Llama-arch
    rope_interleaved: int = 0     # 0 = split-half/NeoX RoPE (Qwen3, GGML type 2);
                                  # 1 = interleaved/NORM (Llama GGUF, GGML type 0)


class DenseDecoder(Model):
    """Stateful dense-transformer inference handle (shared core).

    Family adapters subclass this and override the family-specific seams:
      * :meth:`bake_and_set_rope` — RoPE table construction (Qwen plain vs llama3).
      * :meth:`_prepare_generate_ids` — per-family generation prelude (e.g. BOS).
      * :meth:`from_spec` extension points via the ``Config`` it builds.
    """

    _repr_fields = (("L", "n_layers"), ("D", "d_model"), ("H", "n_heads"),
                    ("hd", "head_dim"), ("n_int", "n_int"))

    def __init__(self, config: Config | str | None = None):
        if config is None: self.cfg = Config()
        elif isinstance(config, str): self.cfg = type(self)._config_preset(config)
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

    # Adapters that support string presets override this; the base has none.
    @classmethod
    def _config_preset(cls, name: str) -> "Config":
        raise ValueError(f"{cls.__name__} has no string presets")

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
        """Load dense-decoder weights from a GGUF file.

        Tensor enumeration + dtype dispatch goes through the shared
        :class:`SuperKittens.inference.load.weight_loader.WeightLoader`; the
        byte-blob upload routes via ``sk_qwen_load_gguf`` so the C++ launcher
        keeps owning GPU-side reallocation.
        """
        from SuperKittens.inference.load.weight_loader import WeightLoader
        # WHY: opening through WeightLoader gives dtype-validated tensor metadata
        # up front and a single ingest path shared across models.
        with WeightLoader(str(path)) as wl:
            _ = sum(1 for _ in wl.iter_tensors())
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

    def prefill_chunked(self, input_ids, *, chunk_size: int = 0) -> int:
        """Prefill a prompt in fixed-size chunks, carrying KV + position across.

        Routes through ``sk_qwen_prefill_chunked`` so a prompt longer than the
        per-step scratch (sized at ``seq_max``) still prefills: per-step memory
        is bounded to ``chunk_size`` rows, not the full prompt. The chunk
        boundary is numerically transparent (the fixed mha_causal), so the
        returned next token (and ``_last_logits``) match a single forward of the
        same prompt. ``chunk_size`` <= 0 uses ``seq_max``. Does NOT reset; call
        ``reset()`` first for a fresh sequence.
        """
        lib = _load()
        if not hasattr(lib, "sk_qwen_prefill_chunked"):
            raise RuntimeError("libsk.dylib has no sk_qwen_prefill_chunked symbol; rebuild dylib")
        ids = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int32)).reshape(-1)
        out = np.empty((1,), dtype=np.int32)
        cs = int(chunk_size) if chunk_size and chunk_size > 0 else 0
        rc = lib.sk_qwen_prefill_chunked(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_uint32(ids.size),
            ctypes.c_uint32(cs),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc:
            raise RuntimeError(f"sk_qwen_prefill_chunked failed: {rc}")
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

    def _prepare_generate_ids(self, input_ids) -> np.ndarray:
        """Per-family generation prelude. Base: identity (no BOS injection)."""
        return np.ascontiguousarray(np.asarray(input_ids, dtype=np.int32)).reshape(-1)

    def generate(self, input_ids, *, max_new_tokens: int = 64,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k=None, eos_id=None, eos_ids=None,
                 seed: int = 0, sampler=None):
        # WHY: keep the entire per-token decode loop in C when the request is
        # plain greedy. Falls back to the Python base-class loop for sampling.
        ids = self._prepare_generate_ids(input_ids)
        lib = _load()
        greedy = (sampler is None
                  and temperature <= 0.0
                  and (top_p >= 1.0 or top_p <= 0.0)
                  and not top_k)
        if not (greedy and hasattr(lib, "sk_qwen_generate_n") and self.cfg.batch == 1):
            return super().generate(ids, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_p=top_p,
                                    top_k=top_k, eos_id=eos_id, eos_ids=eos_ids,
                                    seed=seed, sampler=sampler)
        stops = set()
        if eos_ids:
            stops |= {int(x) for x in eos_ids}
        if eos_id is not None:
            stops.add(int(eos_id))
        if not stops and self.tokenizer is not None:
            t_eos = getattr(self.tokenizer, "eos_ids", None)
            if t_eos: stops |= {int(x) for x in t_eos}
        eos_single = int(next(iter(stops))) if len(stops) == 1 else -1

        ids = np.ascontiguousarray(ids).reshape(-1)
        out = np.empty((int(max_new_tokens),), dtype=np.int32)
        self.reset()
        n = lib.sk_qwen_generate_n(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_uint32(ids.size),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_uint32(int(max_new_tokens)),
            ctypes.c_int32(eos_single))
        if n < 0:
            raise RuntimeError(f"sk_qwen_generate_n failed: {n}")
        toks = out[:n].tolist()
        # Multi-stop: enforce in Python by truncating at first hit.
        if len(stops) > 1:
            for i, t in enumerate(toks):
                if t in stops:
                    toks = toks[:i+1]
                    break
        self._last_token = toks[-1] if toks else None
        return toks

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
        """Build plain RoPE cos/sin tables from cfg and upload to the handle.

        Families with a RoPE-frequency rescale (e.g. Nemotron's llama3 scaling)
        override this seam.
        """
        half = self.cfg.head_dim // 2
        theta = self.cfg.rope_freq_base
        inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        pos = np.arange(self.cfg.cache_max, dtype=np.float64)
        angles = np.outer(pos, inv_freq)
        cos = np.cos(angles).astype(np.float16).copy()
        sin = np.sin(angles).astype(np.float16).copy()
        self.set_rope_tables(cos, sin)

    @classmethod
    def from_spec(cls, spec, **overrides) -> "DenseDecoder":
        """Build a dense-decoder model from a central-registry ModelSpec.

        Reads ``config.json`` from ``spec.weight_dir`` (falling back to
        ``spec.dims`` if absent), constructs a :class:`Config`, loads the
        canonical artifact (GGUF if ``spec.gguf_name`` is set; otherwise
        safetensors), bakes RoPE tables, and attaches the tokenizer. Adapters
        pass family-specific config via ``overrides`` (e.g. ``use_qk_norm=0``).
        """
        import json
        sk_root = Path(__file__).resolve().parents[3]
        snap = Path(overrides.pop("snapshot", None)
                    or (sk_root / "SuperKittens" / "model_weights" / spec.weight_dir))

        # Build Config: prefer on-disk config.json, fall back to spec.dims.
        d = dict(spec.dims)
        cfg_json = snap / "config.json"
        if cfg_json.exists():
            j = json.loads(cfg_json.read_text())
            d.update(
                n_layers   = j.get("num_hidden_layers", d.get("n_layers")),
                d_model    = j.get("hidden_size",     d.get("d_model")),
                n_heads    = j.get("num_attention_heads", d.get("n_heads")),
                n_kv_heads = j.get("num_key_value_heads", d.get("n_kv_heads")),
                head_dim   = j.get("head_dim", d.get("head_dim",
                              (j.get("hidden_size", 0) // max(j.get("num_attention_heads", 1), 1)))),
                n_int      = j.get("intermediate_size", d.get("n_int")),
                vocab_size = j.get("vocab_size",   d.get("vocab_size")),
                eps        = j.get("rms_norm_eps", d.get("eps", 1e-6)),
                rope_freq_base = j.get("rope_theta", d.get("rope_freq_base", 1_000_000.0)),
                tie_word_embeddings = int(j.get("tie_word_embeddings", d.get("tie_word_embeddings", 1))),
            )
        # seq_max / cache_max have sensible Config defaults; allow overrides.
        # Track whether the caller pinned cache_max so the memory-aware clamp
        # only kicks in for the default (auto) case.
        cache_max_pinned = "cache_max" in overrides
        for k in ("seq_max", "cache_max"):
            if k in overrides:
                d[k] = overrides.pop(k)
        # Filter d to Config fields only.
        from dataclasses import fields
        allowed = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in d.items() if k in allowed and v is not None})
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        # Resolve the canonical GGUF path early so its on-disk size can feed the
        # memory-aware cache_max clamp before the native handle (and its KV
        # cache) is allocated.
        gguf_path = None
        if spec.gguf_name:
            gguf_path = snap / spec.gguf_name
            # Also accept the gguf living one level up alongside snapshot dirs.
            if not gguf_path.exists():
                alt = sk_root / "SuperKittens" / "model_weights" / spec.gguf_name
                if alt.exists():
                    gguf_path = alt

        # Memory-aware cache_max: a 14B-Q4 (~9 GB) at the default cache_max of
        # 32768 OOMs a 16 GB box (KV ~5 GB + weights + scratch). When the caller
        # did NOT pin cache_max, clamp it to the largest value that fits.
        if not cache_max_pinned and gguf_path and gguf_path.exists():
            from SuperKittens.inference.generation import (
                memory_aware_cache_max, _unified_memory_bytes)
            weight_bytes = gguf_path.stat().st_size
            mem = _unified_memory_bytes()
            seq_default = Config.seq_max  # type: ignore[attr-defined]
            if (mem and seq_default == cfg.seq_max
                    and weight_bytes > 0.4 * mem and cfg.seq_max > 2048):
                cfg.seq_max = 2048
            fitted = memory_aware_cache_max(
                requested_cache_max=cfg.cache_max,
                seq_max=cfg.seq_max,
                weight_bytes=weight_bytes,
                n_layers=cfg.n_layers, n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim, d_model=cfg.d_model, n_int=cfg.n_int,
                n_heads=cfg.n_heads, vocab_size=cfg.vocab_size,
                total_mem_bytes=mem if mem else None,
                kv_q8=bool(os.environ.get("SK_KV_Q8")),
            )
            if fitted < cfg.cache_max:
                print(f"[{cls.__name__.lower()}] memory-aware cache_max: {cfg.cache_max} -> {fitted} "
                      f"(weights={weight_bytes/2**30:.1f}GiB, seq_max={cfg.seq_max})")
                cfg.cache_max = fitted
                if cfg.seq_max > fitted:
                    cfg.seq_max = fitted

        m = cls(cfg)

        if gguf_path and gguf_path.exists():
            m.load_gguf(str(gguf_path))
        else:
            if not snap.exists():
                raise FileNotFoundError(f"snapshot dir not found: {snap}")
            idx = snap / "model.safetensors.index.json"
            single = snap / "model.safetensors"
            target = idx if idx.exists() else single
            if not target.exists():
                raise FileNotFoundError(f"no safetensors in {snap}")
            rc = _load().sk_qwen_load_safetensors(m._h, str(target).encode())
            if rc:
                raise RuntimeError(f"sk_qwen_load_safetensors failed: {rc}")

        m.bake_and_set_rope()
        m._attach_tokenizer(spec, snap)
        return m

    def _attach_tokenizer(self, spec, snap: Path) -> None:
        if not spec.tokenizer_family:
            return
        try:
            from SuperKittens.models.load.tokenizer import Tokenizer
            json_path = snap / "tokenizer.json"
            sp_path   = snap / "tokenizer.model"
            if json_path.exists():
                self.tokenizer = Tokenizer.from_hf_json(str(json_path), family=spec.tokenizer_family)
            elif sp_path.exists():
                self.tokenizer = Tokenizer.from_sentencepiece(str(sp_path), family=spec.tokenizer_family)
            else:
                print(f"[{type(self).__name__.lower()}] no tokenizer.json or tokenizer.model in {snap}")
        except Exception as e:
            print(f"[{type(self).__name__.lower()}] tokenizer attach failed: {e}")

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
