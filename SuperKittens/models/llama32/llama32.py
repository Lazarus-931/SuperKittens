"""llama32.py — Llama-3.2-{1B,3B}-Instruct adapter over the dense core.

Llama-3.2-{1B,3B}-Instruct is ``model_type=llama`` (LlamaForCausalLM) — the same
dense decoder the shared :class:`DenseDecoder` core drives. The 1B and 3B differ
only in dims (registry rows ``llama-3.2-1b`` / ``llama-3.2-3b``); the adapter is
config-only across both. Configured for the Llama-3.2 arch:

  1. ``use_qk_norm=0`` — Llama has no per-head Q/K RMSNorm.
  2. ``rope_interleaved=1`` — as a Llama-family GGUF, llama.cpp's converter
     permutes q_proj/k_proj for GGML rope type 0 (interleaved/NORM). The core's
     default split-half (NeoX, type 2) kernel would scramble positions on the
     permuted weights (coherent for ~1 token, then degenerates — the Nemotron
     lesson). The interleaved RoPE kernel reproduces HF ``rotate_half``.
  3. :meth:`bake_and_set_rope` applies the Llama-3.2 ``llama3`` piecewise
     inv_freq rescale (theta=500000, factor=32) so the baked cos/sin tables match
     HF ``apply_rotary_pos_emb``. The RoPE kernel only consumes the tables, so no
     kernel change is needed.
  4. ``tie_word_embeddings=1`` — UNLIKE Nemotron/Mistral (untied ``output.weight``),
     Llama-3.2 ties the LM head to the input embedding. The shared core already
     handles the tied case: the C++ launcher reads ``token_embd.weight`` as the LM
     head when ``tie_word_embeddings=1`` (and leaves ``w_lm_head`` null). No
     plumbing change — the tie flag is a generic core config field.
  5. :meth:`_prepare_generate_ids` prepends ``<|begin_of_text|>`` — Llama-3
     completion is BOS-sensitive (without it the model collapses to repeats).

Llama-3.2-3B is its OWN adapter (a :class:`DenseDecoder`, NOT a ``Nemotron`` or
``Qwen``): it shares the launcher/kernels but carries its own model identity,
loader config, RoPE bake, and generation prelude.

Usage:
    import SuperKittens as sk
    m = sk.load("llama-3.2-3b")
    print(m.chat("The capital of France is"))
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from SuperKittens.models.dense.dense_decoder import DenseDecoder


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
    scaled = inv_freq / factor
    out = np.where(wavelen > low_wavelen, scaled, inv_freq)
    smooth = (orig_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1.0 - smooth) * (inv_freq / factor) + smooth * inv_freq
    is_mid = (~(wavelen < high_wavelen)) & (~(wavelen > low_wavelen))
    out = np.where(is_mid, smoothed, out)
    return out


class Llama32(DenseDecoder):
    """Llama-3.2-{1B,3B}-Instruct handle (own adapter; shared dense core)."""

    # Llama3 RoPE scaling, captured from config.json/spec by from_spec.
    # None → plain RoPE (matches the DenseDecoder default).
    _rope_scaling: dict | None = None

    def bake_and_set_rope(self) -> None:
        half = self.cfg.head_dim // 2
        theta = self.cfg.rope_freq_base
        rs = self._rope_scaling
        if rs and str(rs.get("rope_type", rs.get("type", ""))).lower() == "llama3":
            inv_freq = _llama3_inv_freq(
                self.cfg.head_dim, theta,
                factor=float(rs.get("factor", 32.0)),
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
    def from_spec(cls, spec, **overrides) -> "Llama32":
        """Build from a registry ModelSpec.

        Uses the shared :meth:`DenseDecoder.from_spec` for config/GGUF/tokenizer,
        forcing ``use_qk_norm=0`` (Llama arch) and ``rope_interleaved=1`` (Llama
        GGUF NORM rope), and capturing the ``llama3`` rope_scaling block so
        :meth:`bake_and_set_rope` rescales inv_freq. ``tie_word_embeddings``
        comes from config.json (true for Llama-3.2) and the shared core ties
        the LM head to ``token_embd.weight``.
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
        # Llama GGUFs store Q/K permuted for GGML rope type 0 (interleaved/NORM);
        # select the interleaved kernel (see module docstring + the Nemotron note).
        overrides.setdefault("rope_interleaved", 1)
        m = super().from_spec(spec, **overrides)
        m._rope_scaling = rope_scaling
        # DenseDecoder.from_spec already baked plain RoPE; re-bake with llama3.
        m.bake_and_set_rope()
        return m

    def _bos_id(self) -> int:
        # Llama-3 base completion is BOS-sensitive: without <|begin_of_text|>
        # (128000) the model collapses to garbage. Prefer the tokenizer's
        # resolved bos_id, fall back to the canonical id.
        b = getattr(self.tokenizer, "bos_id", None) if self.tokenizer else None
        return int(b) if b is not None else 128000

    def _prepare_generate_ids(self, input_ids) -> np.ndarray:
        # Llama-3 completion is BOS-sensitive: prepend <|begin_of_text|> unless
        # the prompt already starts with it. chat() routes through generate too,
        # so templated turns also get the leading BOS.
        ids = list(np.asarray(input_ids, dtype=np.int64).reshape(-1))
        bos = self._bos_id()
        if not ids or ids[0] != bos:
            ids = [bos] + ids
        return np.array(ids, dtype=np.int32)
