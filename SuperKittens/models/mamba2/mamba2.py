"""SuperKittens Mamba 2 Python wrapper.

Mirrors `SuperKittens/models/qwen/qwen.py`. Currently only handles weight load
and metadata; `forward` is wired through the C ABI but raises NotImplementedError
until the SSD kernel rewrite lands (see `models/mamba2/STATUS.md`).
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from SuperKittens.inference.c_binder import bind
from SuperKittens.inference.generation import Model


@dataclass
class Mamba2Config:
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
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    tie_word_embeddings: int = 1

    @classmethod
    def from_hf_json(cls, path: str | os.PathLike) -> "Mamba2Config":
        j = json.loads(Path(path).read_text())
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
            time_step_min=j.get("time_step_min", 0.001),
            time_step_max=j.get("time_step_max", 0.1),
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
        ("time_step_min", ctypes.c_float),
        ("time_step_max", ctypes.c_float),
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
    }

    def __init__(self, cfg: Mamba2Config, lib_path: str | None = None):
        self.cfg = cfg
        if lib_path is not None:
            os.environ.setdefault("SK_DYLIB", lib_path)
        self._lib = bind("mamba2", self._ABI)
        self._destroy_fn = self._lib.sk_mamba2_destroy

        c_cfg = _CConfig(
            cfg.batch, cfg.seq_max, cfg.n_layers, cfg.d_model, cfg.intermediate,
            cfg.n_heads, cfg.head_dim, cfg.state_size, cfg.n_groups, cfg.conv_kernel,
            cfg.chunk_size, cfg.vocab_size, cfg.rms_eps,
            cfg.time_step_min, cfg.time_step_max, cfg.tie_word_embeddings,
        )
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


# Registry entry — kept import-light.
SPEC = {
    "mamba2-130m": {
        "hf_repo": "AntonV/mamba2-130m-hf",
        "config_cls": Mamba2Config,
        "model_cls": Mamba2Model,
    },
}
