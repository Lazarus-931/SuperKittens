"""granite.py — IBM Granite-4.x hybrid (interleaved mamba2 + attention) adapter.

Drives the sk_granite_* C-ABI launcher: a hybrid stack where per-layer type
comes from GGUF metadata (head_count_kv 0 = mamba2, >0 = attention), every
layer carries a dense SwiGLU FFN, attention is NoPE with the granite
attention_multiplier, and embeddings/residuals/logits carry granite's scalar
multipliers. Reuses the mamba2 family kernels and the shared dense kernels.
"""
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SuperKittens.inference.generation import Model
from SuperKittens.inference.c_binder import bind, optional, CtypesConfig


class _Config(ctypes.Structure):
    _fields_ = [
        ("batch",        ctypes.c_uint32),
        ("seq_max",      ctypes.c_uint32),
        ("cache_max",    ctypes.c_uint32),
        ("n_layers",     ctypes.c_uint32),
        ("d_model",      ctypes.c_uint32),
        ("n_heads",      ctypes.c_uint32),
        ("n_kv_heads",   ctypes.c_uint32),
        ("head_dim",     ctypes.c_uint32),
        ("n_int",        ctypes.c_uint32),
        ("d_inner",      ctypes.c_uint32),
        ("ssm_n_heads",  ctypes.c_uint32),
        ("ssm_head_dim", ctypes.c_uint32),
        ("ssm_state",    ctypes.c_uint32),
        ("ssm_n_groups", ctypes.c_uint32),
        ("ssm_conv",     ctypes.c_uint32),
        ("vocab_size",   ctypes.c_uint32),
        ("eps",             ctypes.c_float),
        ("embedding_scale", ctypes.c_float),
        ("residual_scale",  ctypes.c_float),
        ("attention_scale", ctypes.c_float),
        ("logit_scale",     ctypes.c_float),
    ]


GRANITE_ABI = {
    "create":          ([ctypes.POINTER(_Config)], ctypes.c_void_p),
    "load_gguf":       ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
    "forward":         ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                         ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)], ctypes.c_int),
    "generate_n":      ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                         ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),
                         ctypes.c_uint32, ctypes.c_int32], ctypes.c_int),
    "get_last_logits": ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
    "get_pos":         optional([ctypes.c_void_p], ctypes.c_uint32),
    "reset":           ([ctypes.c_void_p], None),
    "destroy":         ([ctypes.c_void_p], None),
    # batched-lane serving (older dylibs lack these; methods raise cleanly)
    "forward_batched": optional([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                 ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)],
                                ctypes.c_int),
    "prefill_batched": optional([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                 ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)],
                                ctypes.c_int),
    "get_logits_row":  optional([ctypes.c_void_p, ctypes.c_uint32,
                                 ctypes.c_void_p], ctypes.c_int),
}

_lib = None
def _load():
    global _lib
    if _lib is None:
        _lib = bind("granite", GRANITE_ABI)
    return _lib


@dataclass
class Config(CtypesConfig):
    # granite-4.0-h-1b defaults (verified against the GGUF metadata).
    n_layers: int     = 40
    d_model: int      = 1536
    n_heads: int      = 12
    n_kv_heads: int   = 4
    head_dim: int     = 128
    n_int: int        = 4096
    d_inner: int      = 3072
    ssm_n_heads: int  = 48
    ssm_head_dim: int = 64
    ssm_state: int    = 128
    ssm_n_groups: int = 1
    ssm_conv: int     = 4
    vocab_size: int   = 100352
    eps: float        = 1e-5
    embedding_scale: float = 12.0
    residual_scale: float  = 0.22
    attention_scale: float = 0.0078125
    logit_scale: float     = 6.0
    batch: int        = 1
    seq_max: int      = 1024
    cache_max: int    = 4096


class Granite(Model):
    _repr_fields = (("L", "n_layers"), ("D", "d_model"), ("E", "d_inner"),
                    ("ssmH", "ssm_n_heads"), ("attnH", "n_heads"))

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        lib = _load()
        self._destroy_fn = lib.sk_granite_destroy
        self._cstruct = self.cfg.to_c(_Config)
        self._h = lib.sk_granite_create(ctypes.byref(self._cstruct))
        if not self._h:
            raise RuntimeError("sk_granite_create failed (missing PSO?)")
        self._last_token = None
        self.tokenizer = None
        self.vocab_size = self.cfg.vocab_size

    def load_gguf(self, path: str) -> None:
        rc = _load().sk_granite_load_gguf(self._h, str(path).encode())
        if rc:
            raise RuntimeError(f"sk_granite_load_gguf failed: {rc}")

    def reset(self) -> None:
        _load().sk_granite_reset(self._h)
        self._last_token = None

    def _forward(self, input_ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int32)).reshape(-1)
        out = np.empty((1,), dtype=np.int32)
        rc = _load().sk_granite_forward(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ids.size,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc:
            raise RuntimeError(f"sk_granite_forward failed: {rc}")
        self._last_token = int(out[0])
        return out

    def forward(self, input_ids) -> int:
        return int(self._forward(input_ids)[0])

    def _last_logits(self) -> np.ndarray:
        out = np.empty((self.cfg.vocab_size,), dtype=np.float16)
        rc = _load().sk_granite_get_last_logits(self._h, out.ctypes.data)
        if rc:
            raise RuntimeError(f"sk_granite_get_last_logits failed: {rc}")
        return out

    # ── batched-lane serving (cfg.batch = N lockstep lanes) ──────────────

    def logits_row(self, lane: int) -> np.ndarray:
        lib = _load()
        if not hasattr(lib, "sk_granite_get_logits_row"):
            raise RuntimeError("dylib lacks sk_granite_get_logits_row")
        out = np.empty((self.cfg.vocab_size,), dtype=np.float16)
        rc = lib.sk_granite_get_logits_row(self._h, lane, out.ctypes.data)
        if rc:
            raise RuntimeError(f"sk_granite_get_logits_row failed: {rc}")
        return out

    def forward_batched(self, tokens) -> np.ndarray:
        """One lockstep step: tokens (batch,) -> greedy next tokens (batch,)."""
        lib = _load()
        if not hasattr(lib, "sk_granite_forward_batched"):
            raise RuntimeError("dylib lacks sk_granite_forward_batched")
        ids = np.ascontiguousarray(np.asarray(tokens, dtype=np.int32)).reshape(-1)
        if ids.size != self.cfg.batch:
            raise ValueError(f"need {self.cfg.batch} tokens, got {ids.size}")
        out = np.empty((self.cfg.batch,), dtype=np.int32)
        rc = lib.sk_granite_forward_batched(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc:
            raise RuntimeError(f"sk_granite_forward_batched failed: {rc}")
        return out

    def prefill_batched(self, prompts) -> np.ndarray:
        """All N equal-length prompts in ONE batch=N forward (fresh handle
        state required); returns per-lane greedy first tokens (batch,)."""
        lib = _load()
        if not hasattr(lib, "sk_granite_prefill_batched"):
            raise RuntimeError("dylib lacks sk_granite_prefill_batched")
        rows = [np.asarray(p, dtype=np.int32).reshape(-1) for p in prompts]
        if len(rows) != self.cfg.batch:
            raise ValueError(f"need {self.cfg.batch} prompts, got {len(rows)}")
        seq = rows[0].size
        if any(r.size != seq for r in rows):
            raise ValueError("prefill_batched requires equal-length prompts")
        flat = np.ascontiguousarray(np.stack(rows).reshape(-1))
        out = np.empty((self.cfg.batch,), dtype=np.int32)
        rc = lib.sk_granite_prefill_batched(
            self._h,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), seq,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc:
            raise RuntimeError(f"sk_granite_prefill_batched failed: {rc}")
        return out

    def generate_batched(self, prompts, *, max_new_tokens: int = 64,
                         batched_prefill: bool = False):
        """Greedy lockstep serving: N equal-length prompts -> N token lists.
        Prefill is token-by-token lockstep (default) or one batched forward;
        decode is sk_granite_forward_batched either way. Resets the handle."""
        rows = [np.asarray(p, dtype=np.int32).reshape(-1) for p in prompts]
        if len(rows) != self.cfg.batch:
            raise ValueError(f"need {self.cfg.batch} prompts, got {len(rows)}")
        seq = rows[0].size
        if any(r.size != seq for r in rows):
            raise ValueError("generate_batched requires equal-length prompts")
        self.reset()
        if batched_prefill:
            cur = self.prefill_batched(rows)
        else:
            mat = np.stack(rows)
            for t in range(seq):
                cur = self.forward_batched(mat[:, t])
        outs = [[int(c)] for c in cur]
        for _ in range(int(max_new_tokens) - 1):
            cur = self.forward_batched(cur)
            for lane, c in enumerate(cur):
                outs[lane].append(int(c))
        return outs

    def generate(self, input_ids, *, max_new_tokens: int = 64,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k=None, eos_id=None, eos_ids=None,
                 seed: int = 0, sampler=None):
        greedy = (sampler is None and temperature <= 0.0
                  and (top_p >= 1.0 or top_p <= 0.0) and not top_k)
        if not greedy:
            return super().generate(input_ids, max_new_tokens=max_new_tokens,
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
            if t_eos:
                stops |= {int(x) for x in t_eos}
        eos_single = int(next(iter(stops))) if len(stops) == 1 else -1

        ids = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int32)).reshape(-1)
        out = np.empty((int(max_new_tokens),), dtype=np.int32)
        self.reset()
        n = _load().sk_granite_generate_n(
            self._h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_uint32(ids.size),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_uint32(int(max_new_tokens)),
            ctypes.c_int32(eos_single))
        if n < 0:
            raise RuntimeError(f"sk_granite_generate_n failed: {n}")
        toks = out[:n].tolist()
        if len(stops) > 1:
            for i, t in enumerate(toks):
                if t in stops:
                    toks = toks[:i + 1]
                    break
        self._last_token = toks[-1] if toks else None
        return toks

    @classmethod
    def from_spec(cls, spec, **overrides) -> "Granite":
        sk_root = Path(__file__).resolve().parents[3]
        snap = Path(overrides.pop("snapshot", None)
                    or (sk_root / "SuperKittens" / "model_weights" / spec.weight_dir))

        from dataclasses import fields
        allowed = {f.name for f in fields(Config)}
        d = {k: v for k, v in dict(spec.dims).items() if k in allowed}
        cfg = Config(**d)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        gguf_path = snap / spec.gguf_name if spec.gguf_name else None
        if not gguf_path or not gguf_path.exists():
            raise FileNotFoundError(f"granite GGUF not found: {gguf_path}")

        m = cls(cfg)
        m.load_gguf(str(gguf_path))
        m._attach_tokenizer(spec, snap)
        return m

    def _attach_tokenizer(self, spec, snap: Path) -> None:
        if not spec.tokenizer_family:
            return
        try:
            from SuperKittens.models.load.tokenizer import Tokenizer
            json_path = snap / "tokenizer.json"
            if json_path.exists():
                self.tokenizer = Tokenizer.from_hf_json(str(json_path),
                                                        family=spec.tokenizer_family)
            else:
                print(f"[granite] no tokenizer.json in {snap}")
        except Exception as e:
            print(f"[granite] tokenizer attach failed: {e}")
