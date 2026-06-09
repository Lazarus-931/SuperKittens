"""mistral.py — Mistral-7B-Instruct-v0.3 adapter over the dense core.

Mistral-7B is ``model_type=mistral`` (MistralForCausalLM) — a Llama-style dense
decoder the shared :class:`DenseDecoder` core already drives. Configured for the
Mistral arch:

  1. ``use_qk_norm=0`` — Mistral has no per-head Q/K RMSNorm.
  2. ``rope_interleaved=1`` — as a Llama-family GGUF, llama.cpp's converter
     permutes q_proj/k_proj for GGML rope type 0 (interleaved/NORM). The core's
     default split-half (NeoX, type 2) kernel would scramble positions on the
     permuted weights (coherent for ~1 token, then degenerates — the Nemotron
     lesson). The interleaved RoPE kernel reproduces HF ``rotate_half``.
  3. Plain RoPE with ``theta=1e6`` (no llama3 frequency rescale) — so the base
     :meth:`DenseDecoder.bake_and_set_rope` is correct as-is; no override needed.
  4. Untied LM head (``output.weight``) — ``tie_word_embeddings=0``.

Unlike Llama-3/Nemotron, Mistral generation is not BOS-prefix-sensitive in a way
that needs a generate-time injection: the chat path prepends ``<s>`` via
``Tokenizer.chat(bos=True)``. v0.3 carries no GGUF-side sliding-window restriction
on the decode path, so attention runs full-causal like the other dense families.

Mistral is its OWN adapter (a :class:`DenseDecoder`, NOT a ``Qwen`` or
``Nemotron``): it shares the launcher/kernels but carries its own model identity
and loader config.

Usage:
    import SuperKittens as sk
    m = sk.load("mistral-7b-v0.3")
    print(m.chat("The capital of France is"))
"""
from __future__ import annotations

from SuperKittens.models.dense.dense_decoder import DenseDecoder


class Mistral(DenseDecoder):
    """Mistral-7B-Instruct-v0.3 handle (own adapter; shared dense core)."""

    @classmethod
    def from_spec(cls, spec, **overrides) -> "Mistral":
        """Build from a registry ModelSpec.

        Uses the shared :meth:`DenseDecoder.from_spec` for config/GGUF/tokenizer,
        forcing ``use_qk_norm=0`` (Mistral arch) and ``rope_interleaved=1`` (Llama
        GGUF NORM rope). RoPE is plain (theta=1e6); the base bake is correct.
        """
        overrides.setdefault("use_qk_norm", 0)
        overrides.setdefault("rope_interleaved", 1)
        return super().from_spec(spec, **overrides)
