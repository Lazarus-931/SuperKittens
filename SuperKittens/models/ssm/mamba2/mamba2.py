"""SuperKittens Mamba 2 Python wrapper.

Mirrors `SuperKittens/models/qwen/qwen.py`. Currently only handles weight load
and metadata; `forward` is wired through the C ABI but raises NotImplementedError
until the SSD kernel rewrite lands (see `models/ssm/mamba2/STATUS.md`).
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from SuperKittens.inference.c_binder import bind, optional, CtypesConfig
from SuperKittens.inference.generation import Model


@dataclass
class Mamba2Config(CtypesConfig):
    batch: int = 1
    seq_max: int = 2048
    n_layers: int = 24
    d_model: int = 768
    intermediate: int = 1536       # E = expand * d_model
    n_heads: int = 24              # H
    head_dim: int = 64             # P
    state_size: int = 128          # N
    n_groups: int = 1              # G
    conv_kernel: int = 4
    chunk_size: int = 256
    vocab_size: int = 50288
    rms_eps: float = 1e-5
    # Runtime dt clamp = HF time_step_limit (NOT time_step_{min,max}, which only
    # bound dt_bias init). HF default (0.0, inf): lower no-op, no upper clamp.
    dt_limit_min: float = 0.0
    dt_limit_max: float = float("inf")
    tie_word_embeddings: int = 1

    @classmethod
    def from_hf_json(cls, path: str | os.PathLike) -> "Mamba2Config":
        j = json.loads(Path(path).read_text())
        lim = j.get("time_step_limit", [0.0, float("inf")])
        return cls(
            n_layers=j["num_hidden_layers"],
            d_model=j["hidden_size"],
            intermediate=j.get("intermediate_size", j["hidden_size"] * j.get("expand", 2)),
            n_heads=j["num_heads"],
            head_dim=j["head_dim"],
            state_size=j["state_size"],
            n_groups=j.get("n_groups", 1),
            conv_kernel=j.get("conv_kernel", 4),
            chunk_size=j.get("chunk_size", 256),
            vocab_size=j["vocab_size"],
            rms_eps=j.get("layer_norm_epsilon", j.get("rms_norm_eps", 1e-5)),
            dt_limit_min=float(lim[0]),
            dt_limit_max=float(lim[1]),
            tie_word_embeddings=int(j.get("tie_word_embeddings", True)),
        )


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("batch",         ctypes.c_uint32),
        ("seq_max",       ctypes.c_uint32),
        ("n_layers",      ctypes.c_uint32),
        ("d_model",       ctypes.c_uint32),
        ("intermediate",  ctypes.c_uint32),
        ("n_heads",       ctypes.c_uint32),
        ("head_dim",      ctypes.c_uint32),
        ("state_size",    ctypes.c_uint32),
        ("n_groups",      ctypes.c_uint32),
        ("conv_kernel",   ctypes.c_uint32),
        ("chunk_size",    ctypes.c_uint32),
        ("vocab_size",    ctypes.c_uint32),
        ("rms_eps",       ctypes.c_float),
        ("dt_limit_min",  ctypes.c_float),
        ("dt_limit_max",  ctypes.c_float),
        ("tie_word_embeddings", ctypes.c_uint32),
    ]


class Mamba2Model(Model):
    """ctypes binding for libSuperKittens Mamba 2 C ABI."""

    _ABI = {
        "create":           ([ctypes.POINTER(_CConfig)], ctypes.c_void_p),
        "load_safetensors": ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
        "forward":          ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                              ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
        "reset":            ([ctypes.c_void_p], None),
        "destroy":          ([ctypes.c_void_p], None),
        "dump_layer":       ([ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
        "get_last_logits":  ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
        # Batched-decode serving (optional — present only in batched dylibs).
        "reset_lane":       optional([ctypes.c_void_p, ctypes.c_uint32], None),
        "prefill_lane":     optional([ctypes.c_void_p, ctypes.c_uint32,
                                      ctypes.POINTER(ctypes.c_int), ctypes.c_uint32,
                                      ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
        "prefill_batched":  optional([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
        "decode_batched":   optional([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    }

    def __init__(self, cfg: Mamba2Config, lib_path: str | None = None):
        self.cfg = cfg
        if lib_path is not None:
            os.environ.setdefault("SK_DYLIB", lib_path)
        self._lib = bind("mamba2", self._ABI)
        self._destroy_fn = self._lib.sk_mamba2_destroy

        c_cfg = cfg.to_c(_CConfig)
        self._h = self._lib.sk_mamba2_create(ctypes.byref(c_cfg))
        if not self._h:
            raise RuntimeError("sk_mamba2_create returned NULL")

    def load_safetensors(self, path: str | os.PathLike) -> None:
        rc = self._lib.sk_mamba2_load_safetensors(self._h, str(path).encode())
        if rc != 0:
            raise RuntimeError(f"sk_mamba2_load_safetensors rc={rc}")

    def forward(self, input_ids):
        ids = (ctypes.c_int * len(input_ids))(*input_ids)
        out = ctypes.c_int(0)
        rc = self._lib.sk_mamba2_forward(self._h, ids, len(input_ids), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"sk_mamba2_forward rc={rc}")
        return out.value

    def _forward(self, input_ids):
        import numpy as np
        return np.array([self.forward(list(int(i) for i in np.asarray(input_ids).reshape(-1)))],
                        dtype=np.int32)

    def _last_logits(self):
        return self.get_last_logits()

    def generate(self, input_ids, *, max_new_tokens: int = 64, **kw):
        """Greedy/sampled generation, O(1) per decode step.

        Prefill the prompt once (captures per-layer conv_state + ssm_state),
        then feed one token per step: conv1d_silu_step rolls the carried
        (K-1)-token conv window and mamba2_ssd carries the SSM state, so no
        re-prefill is needed. Token-for-token identical to the prior O(T^2)
        re-prefill path and to HF torch_forward.
        """
        import numpy as np
        ids = [int(i) for i in np.asarray(input_ids, dtype=np.int32).reshape(-1)]
        rng = np.random.default_rng(kw.get("seed", 0))
        temperature = kw.get("temperature", 0.0)
        top_p = kw.get("top_p", 1.0)
        top_k = kw.get("top_k", None)
        greedy = temperature <= 0.0 and (top_p >= 1.0 or top_p <= 0.0) and not top_k
        stops: set = set()
        if kw.get("eos_ids"):
            stops |= {int(x) for x in kw["eos_ids"]}
        if kw.get("eos_id") is not None:
            stops.add(int(kw["eos_id"]))

        self.reset()
        arg = self.forward(ids)                 # prefill
        nxt = int(arg) if greedy else self._sample(
            self.get_last_logits(), temperature, top_p, top_k, rng)
        out: list[int] = [nxt]
        if nxt in stops:
            return out
        for _ in range(max_new_tokens - 1):
            arg = self.forward([nxt])           # O(1) decode step
            nxt = int(arg) if greedy else self._sample(
                self.get_last_logits(), temperature, top_p, top_k, rng)
            out.append(nxt)
            if nxt in stops:
                break
        return out

    def prefill_batched(self, prompts):
        """Prefill N lanes' EQUAL-LENGTH prompts in ONE batch=N forward.

        One weight read for all N prompts (vs N sequential prefill_lane
        forwards) and N× the threadgroups for the occupancy-starved SSD scan.
        Resets each lane's state, then returns the per-lane greedy first
        tokens; lane states are left ready for sk_mamba2_decode_batched.
        """
        import ctypes as C
        N = len(prompts)
        if N == 0:
            return []
        if N > self.cfg.batch:
            raise ValueError(f"N={N} > cfg.batch={self.cfg.batch}")
        seq = len(prompts[0])
        if any(len(p) != seq for p in prompts):
            raise ValueError("prefill_batched requires equal-length prompts")
        if not hasattr(self._lib, "sk_mamba2_prefill_batched"):
            raise RuntimeError("dylib missing batched-prefill ABI")
        for lane in range(N):
            self._lib.sk_mamba2_reset_lane(self._h, lane)
        flat = (C.c_int * (N * seq))(*[int(t) for p in prompts for t in p])
        outb = (C.c_int * N)()
        rc = self._lib.sk_mamba2_prefill_batched(self._h, flat, seq, N, outb)
        if rc != 0:
            raise RuntimeError(f"prefill_batched rc={rc}")
        return [int(outb[i]) for i in range(N)]

    def generate_batched(self, prompts, *, max_new_tokens: int = 64,
                         batched_prefill: bool = False):
        """Greedy lockstep batched decode of N requests (serving path).

        Each prompt is prefilled into its own lane's conv+ssm state, then the
        decode runs all N lanes in lockstep: every step projects N rows through
        ONE weight read (in/out_proj GEMM at M=N) — the weight-read amortization
        that makes aggregate tok/s scale with N. Per-lane SSM state isolation
        means each lane's tokens match its single-stream M=1 run.

        prompts: list of token-id lists (len <= cfg.batch). Returns list of
        per-lane generated token-id lists (greedy). batched_prefill=True runs
        all prompts in one batch=N forward (equal lengths only).
        """
        import ctypes as C
        N = len(prompts)
        if N == 0:
            return []
        if N > self.cfg.batch:
            raise ValueError(f"N={N} > cfg.batch={self.cfg.batch}")
        if not hasattr(self._lib, "sk_mamba2_decode_batched"):
            raise RuntimeError("dylib missing batched-decode ABI")

        cur = [0] * N
        out = [[] for _ in range(N)]
        if batched_prefill:
            cur = self.prefill_batched(prompts)
            for lane in range(N):
                out[lane].append(cur[lane])
        else:
            # Per-lane prefill: each request's prompt -> its lane's state.
            for lane, ids in enumerate(prompts):
                self._lib.sk_mamba2_reset_lane(self._h, lane)
                arr = (C.c_int * len(ids))(*[int(x) for x in ids])
                o = C.c_int(0)
                rc = self._lib.sk_mamba2_prefill_lane(self._h, lane, arr, len(ids), C.byref(o))
                if rc != 0:
                    raise RuntimeError(f"prefill_lane({lane}) rc={rc}")
                cur[lane] = int(o.value)
                out[lane].append(cur[lane])

        # Lockstep batched decode.
        inbuf  = (C.c_int * N)()
        outbuf = (C.c_int * N)()
        for _ in range(max_new_tokens - 1):
            for lane in range(N):
                inbuf[lane] = cur[lane]
            rc = self._lib.sk_mamba2_decode_batched(self._h, inbuf, N, outbuf)
            if rc != 0:
                raise RuntimeError(f"decode_batched rc={rc}")
            for lane in range(N):
                cur[lane] = int(outbuf[lane])
                out[lane].append(cur[lane])
        return out

    def dump(self, tag: str):
        """Return numpy fp16/fp32 array (best-effort sized to known buffer)."""
        import numpy as np
        # Size lookup table for known tags.
        c = self.cfg
        T = c.batch * c.seq_max
        sizes = {
            "embed":         (T * c.d_model, np.float16),
            "x":             (T * c.d_model, np.float16),
            "x_norm":        (T * c.d_model, np.float16),
            "in_proj_out":   (T * (2 * c.intermediate + 2 * c.n_groups * c.state_size + c.n_heads), np.float16),
            "z":             (T * c.intermediate, np.float16),
            "xBC":           (T * (c.intermediate + 2 * c.n_groups * c.state_size), np.float16),
            "dt_raw":        (T * c.n_heads, np.float16),
            "xBC_post":      (T * (c.intermediate + 2 * c.n_groups * c.state_size), np.float16),
            "ssd_out":       (T * c.intermediate, np.float16),
            "gated":         (T * c.intermediate, np.float16),
            "out_proj_out":  (T * c.d_model, np.float16),
            "logits":        (T * c.vocab_size, np.float16),
        }
        if tag.startswith("ssm_state."):
            n = c.batch * c.n_heads * c.head_dim * c.state_size
            arr = np.zeros(n, dtype=np.float32)
        elif tag.startswith("conv_state."):
            n = c.batch * (c.conv_kernel - 1) * (c.intermediate + 2 * c.n_groups * c.state_size)
            arr = np.zeros(n, dtype=np.float16)
        else:
            n, dt = sizes[tag]
            arr = np.zeros(n, dtype=dt)
        rc = self._lib.sk_mamba2_dump_layer(self._h, tag.encode(), arr.ctypes.data, arr.nbytes)
        if rc != 0:
            raise RuntimeError(f"sk_mamba2_dump_layer({tag!r}) rc={rc}")
        return arr

    def get_last_logits(self):
        import numpy as np
        v = np.zeros(self.cfg.vocab_size, dtype=np.float16)
        rc = self._lib.sk_mamba2_get_last_logits(self._h, v.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"sk_mamba2_get_last_logits rc={rc}")
        return v

    def reset(self) -> None:
        self._lib.sk_mamba2_reset(self._h)

    @classmethod
    def from_spec(cls, spec, **overrides) -> "Mamba2Model":
        """Build a Mamba2 model from a central-registry ModelSpec.

        Mamba2 has no RoPE (it's a state-space model). The pre-instruct
        130m checkpoint also has no chat template, so tokenizer attach is
        best-effort and gated on ``spec.tokenizer_family``.
        """
        sk_root = Path(__file__).resolve().parents[3]
        snap = Path(overrides.pop("snapshot", None)
                    or (sk_root / "SuperKittens" / "model_weights" / spec.weight_dir))
        cfg_json = snap / "config.json"
        if cfg_json.exists():
            cfg = Mamba2Config.from_hf_json(cfg_json)
        else:
            # Fall back to spec.dims (filtered to Mamba2Config fields).
            from dataclasses import fields
            allowed = {f.name for f in fields(Mamba2Config)}
            cfg = Mamba2Config(**{k: v for k, v in spec.dims.items() if k in allowed})
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        m = cls(cfg)

        # Weights: prefer index, then single safetensors, then directory itself.
        idx = snap / "model.safetensors.index.json"
        single = snap / "model.safetensors"
        if single.exists():
            m.load_safetensors(single)
        elif idx.exists():
            m.load_safetensors(idx)
        elif snap.exists():
            m.load_safetensors(snap)
        else:
            raise FileNotFoundError(f"snapshot dir not found: {snap}")

        m.tokenizer = None
        if spec.tokenizer_family:
            try:
                from SuperKittens.models.load.tokenizer import Tokenizer
                json_path = snap / "tokenizer.json"
                sp_path = snap / "tokenizer.model"
                if json_path.exists():
                    m.tokenizer = Tokenizer.from_hf_json(str(json_path), family=spec.tokenizer_family)
                elif sp_path.exists():
                    m.tokenizer = Tokenizer.from_sentencepiece(str(sp_path), family=spec.tokenizer_family)
            except Exception as e:
                print(f"[mamba2] tokenizer attach failed: {e}")
        return m


# Registry entry — kept import-light.
SPEC = {
    "mamba2-130m": {
        "hf_repo": "AntonV/mamba2-130m-hf",
        "config_cls": Mamba2Config,
        "model_cls": Mamba2Model,
    },
}
