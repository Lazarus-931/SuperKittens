"""phi4.py — Phi-4-reasoning (14B) adapter over the dense core.

Phi-4-reasoning is ``model_type=phi3`` (Phi3ForCausalLM) — a dense decoder the
shared :class:`DenseDecoder` core already drives. Config (verified against the
HF config.json AND the GGUF header): 40 layers, d_model 5120, 40 heads / 10 KV
heads, head_dim 128, n_int 17920, vocab 100352, rms_eps 1e-5. Configured:

  1. ``use_qk_norm=0`` — Phi-3/4 has no per-head Q/K RMSNorm.
  2. ``rope_interleaved=0`` — phi3-arch GGUFs are NOT q/k-permuted (only the
     llama arch gets the type-0 conversion permute), so the GGUF wants GGML
     rope type 2 (NeoX/split-half), the core's default kernel. This matches HF
     ``rotate_half`` on unpermuted weights — same convention as Qwen3.
  3. Plain RoPE, ``theta=500000``, ``rope_scaling=null``,
     ``partial_rotary_factor=1.0`` — full 128-dim rotary, verified via the
     GGUF's ``phi3.rope.dimension_count=128`` (a partial factor would need a
     kernel change; the core has no support for it). The base
     :meth:`DenseDecoder.bake_and_set_rope` is correct as-is.
  4. Untied LM head (``output.weight``) — ``tie_word_embeddings=0``.
  5. ``attention_bias=false`` — no QKV bias (no bias_add path).
  6. No BOS prepend: the tiktoken-style vocab is GPT-lineage; completion is not
     BOS-sensitive and the chat template starts at ``<|im_start|>``.

ARTIFACT NOTE: phi3-arch GGUFs ship FUSED projections — ``blk.N.attn_qkv``
([Q;K;V] row-concat) and ``blk.N.ffn_up`` holding [gate;up] (2*n_ff rows).
The shared loader wants separate attn_q/k/v + ffn_gate/ffn_up tensors, so the
GGUF is repacked ONCE offline with ``repack_phi3_gguf.py`` (pure row-range byte
split — K-quant rows are independent, so the split is bit-exact, no requant).
The registry's ``gguf_name`` points at the repacked ``*-sk.gguf``.

Usage:
    import SuperKittens as sk
    m = sk.load("phi4-reasoning")
    print(m.chat("Why is the sky blue?"))
"""
from __future__ import annotations

from SuperKittens.models.dense.dense_decoder import DenseDecoder


class Phi4(DenseDecoder):
    """Phi-4-reasoning handle (own adapter; shared dense core)."""

    @classmethod
    def from_spec(cls, spec, **overrides) -> "Phi4":
        """Build from a registry ModelSpec.

        Uses the shared :meth:`DenseDecoder.from_spec` for config/GGUF/tokenizer,
        forcing ``use_qk_norm=0`` (phi3 arch) and ``rope_interleaved=0`` (NeoX,
        unpermuted GGUF). RoPE is plain (theta=5e5); the base bake is correct.
        """
        overrides.setdefault("use_qk_norm", 0)
        overrides.setdefault("rope_interleaved", 0)
        return super().from_spec(spec, **overrides)
