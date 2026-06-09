"""yi.py — Yi-1.5 (01.AI) adapter over the shared dense core.

Yi-1.5 is ``model_type=llama`` (LlamaForCausalLM) — a Llama-style dense decoder
the shared :class:`DenseDecoder` core already drives. A distinct vendor (01.AI),
architecturally the same family as Mistral on this launcher:

  1. ``use_qk_norm=0`` — Yi has no per-head Q/K RMSNorm.
  2. ``rope_interleaved=1`` — as a Llama-family GGUF, llama.cpp's converter
     permutes q_proj/k_proj for GGML rope type 0 (interleaved/NORM). The core's
     default split-half (NeoX, type 2) kernel would scramble positions on the
     permuted weights. The interleaved RoPE kernel reproduces HF rotate_half.
  3. Plain RoPE with theta=5e6 (no llama3 frequency rescale) — so the base
     :meth:`DenseDecoder.bake_and_set_rope` is correct as-is; no override needed.
  4. Untied LM head (output.weight, Q6_K) — ``tie_word_embeddings=0``.
  5. attention_bias=False — no QKV bias (unlike Qwen2.5; no bias_add path).
  6. BOS-sensitive: without <|startoftext|> (id 1) the model degenerates into
     a repeating loop after a coherent first clause (the Llama-3/Nemotron lesson;
     here it survives Q4_K_M but loops). :meth:`_prepare_generate_ids` prepends it.

Chat is ChatML (<|im_start|>/<|im_end|>), same structure as Qwen; the tokenizer
routes through the ``yi`` family (its own 64k SentencePiece-BPE vocab, loaded
from tokenizer.json).

Yi is its OWN adapter (a :class:`DenseDecoder`, NOT a ``Mistral`` or ``Qwen``):
it shares the launcher/kernels but carries its own model identity.

Usage:
    import SuperKittens as sk
    m = sk.load("yi-1.5-6b-chat")
    print(m.chat("The capital of France is"))
"""
from __future__ import annotations

import numpy as np

from SuperKittens.models.dense.dense_decoder import DenseDecoder


class Yi(DenseDecoder):
    """Yi-1.5 handle (own adapter; shared dense core)."""

    @classmethod
    def from_spec(cls, spec, **overrides) -> "Yi":
        """Build from a registry ModelSpec.

        Uses the shared :meth:`DenseDecoder.from_spec` for config/GGUF/tokenizer,
        forcing ``use_qk_norm=0`` (Llama arch) and ``rope_interleaved=1`` (Llama
        GGUF NORM rope). RoPE is plain (theta=5e6); the base bake is correct.
        """
        overrides.setdefault("use_qk_norm", 0)
        overrides.setdefault("rope_interleaved", 1)
        return super().from_spec(spec, **overrides)

    def _bos_id(self) -> int:
        b = getattr(self.tokenizer, "bos_id", None) if self.tokenizer else None
        return int(b) if b is not None else 1  # <|startoftext|> = 1

    def _prepare_generate_ids(self, input_ids) -> np.ndarray:
        # Yi completion is BOS-sensitive: prepend <|startoftext|> unless the prompt
        # already starts with it. Without it the model loops after one clause.
        ids = list(np.asarray(input_ids, dtype=np.int64).reshape(-1))
        bos = self._bos_id()
        if not ids or ids[0] != bos:
            ids = [bos] + ids
        return np.array(ids, dtype=np.int32)
