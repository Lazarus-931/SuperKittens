"""nemotron.py — Llama-3.1-Nemotron-Nano-8B-v1 inference.

Nano-8B is `model_type=llama` (LlamaForCausalLM) — i.e. the Qwen3 dense decoder
minus per-head Q/K-norm, with Llama-3.1 RoPE (theta=500000 + "llama3" frequency
scaling). It reuses the entire Qwen launcher/kernel stack; the only deltas are:

  1. ``use_qk_norm=0`` — Llama has no per-head Q/K RMSNorm (gated in dispatch_layer).
  2. ``bake_and_set_rope`` applies the llama3 piecewise inv_freq rescale so the
     baked cos/sin tables match HF ``apply_rotary_pos_emb`` (the rope kernel only
     consumes the tables, so no kernel change is needed).

Usage:
    import SuperKittens as sk
    m = sk.load("nemotron-nano-8b")
    print(m.chat("The capital of France is"))
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from SuperKittens.models.qwen.qwen import Qwen


def _llama3_inv_freq(head_dim: int, theta: float, *,
                     factor: float, low_freq_factor: float,
                     high_freq_factor: float, orig_ctx: int) -> np.ndarray:
    """HF `_compute_llama3_parameters` inv_freq rescale.

    Frequencies whose wavelength is shorter than ``orig/high`` are untouched;
    longer than ``orig/low`` are divided by ``factor``; in between, a smooth
    interpolation. Returns inv_freq[head_dim/2] (float64).
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / head_dim))
    low_wavelen = orig_ctx / low_freq_factor
    high_wavelen = orig_ctx / high_freq_factor
    wavelen = 2.0 * np.pi / inv_freq
    # > low_wavelen → scale down by factor; < high_wavelen → keep.
    scaled = inv_freq / factor
    out = np.where(wavelen > low_wavelen, scaled, inv_freq)
    # Smooth band between high and low.
    smooth = (orig_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1.0 - smooth) * (inv_freq / factor) + smooth * inv_freq
    is_mid = (~(wavelen < high_wavelen)) & (~(wavelen > low_wavelen))
    out = np.where(is_mid, smoothed, out)
    return out


class Nemotron(Qwen):
    """Llama-3.1-Nemotron-Nano-8B-v1 handle (Qwen launcher, Llama-arch config)."""

    # Llama3 RoPE scaling, captured from config.json by from_spec. None → plain RoPE.
    _rope_scaling: dict | None = None

    def bake_and_set_rope(self) -> None:
        half = self.cfg.head_dim // 2
        theta = self.cfg.rope_freq_base
        rs = self._rope_scaling
        if rs and str(rs.get("rope_type", rs.get("type", ""))).lower() == "llama3":
            inv_freq = _llama3_inv_freq(
                self.cfg.head_dim, theta,
                factor=float(rs.get("factor", 8.0)),
                low_freq_factor=float(rs.get("low_freq_factor", 1.0)),
                high_freq_factor=float(rs.get("high_freq_factor", 4.0)),
                orig_ctx=int(rs.get("original_max_position_embeddings", 8192)),
            )
        else:
            inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        pos = np.arange(self.cfg.cache_max, dtype=np.float64)
        angles = np.outer(pos, inv_freq)
        cos = np.cos(angles).astype(np.float16).copy()
        sin = np.sin(angles).astype(np.float16).copy()
        self.set_rope_tables(cos, sin)

    @classmethod
    def from_spec(cls, spec, **overrides) -> "Nemotron":
        """Build from a registry ModelSpec.

        Reuses Qwen.from_spec for the heavy lifting (config.json → Config, GGUF
        load, tokenizer attach) but forces use_qk_norm=0 and captures the llama3
        rope_scaling block so bake_and_set_rope rescales inv_freq correctly.
        """
        sk_root = Path(__file__).resolve().parents[3]
        snap = Path(overrides.get("snapshot")
                    or (sk_root / "SuperKittens" / "model_weights" / spec.weight_dir))
        rope_scaling = None
        cfg_json = snap / "config.json"
        if cfg_json.exists():
            j = json.loads(cfg_json.read_text())
            rope_scaling = j.get("rope_scaling")
        else:
            rope_scaling = dict(spec.dims.get("rope_scaling")) if spec.dims.get("rope_scaling") else None

        overrides.setdefault("use_qk_norm", 0)
        m = super().from_spec(spec, **overrides)
        m._rope_scaling = rope_scaling
        # Qwen.from_spec already baked plain RoPE; re-bake with llama3 scaling.
        m.bake_and_set_rope()
        return m

    def _bos_id(self) -> int:
        # Llama-3 base completion is BOS-sensitive: without <|begin_of_text|>
        # (128000) the model collapses to garbage. Prefer the tokenizer's
        # resolved bos_id, fall back to the canonical id.
        b = getattr(self.tokenizer, "bos_id", None) if self.tokenizer else None
        return int(b) if b is not None else 128000

    def generate(self, input_ids, **kw):
        # Llama-3 completion is BOS-sensitive: prepend <|begin_of_text|> unless
        # the prompt already starts with it. chat() routes through here too, so
        # templated turns also get the leading BOS.
        ids = list(np.asarray(input_ids, dtype=np.int64).reshape(-1))
        bos = self._bos_id()
        if not ids or ids[0] != bos:
            ids = [bos] + ids
        return super().generate(np.array(ids, dtype=np.int32), **kw)
