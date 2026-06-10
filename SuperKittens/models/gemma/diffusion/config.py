"""config.py — DiffusionGemma family config, mapped from `diffusion-gemma` GGUF
metadata (per-layer kv-head array, sliding_window_pattern, dual head/rope dims,
expert counts, canvas_length, softcap, mask token)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiffusionGemmaConfig:
    n_layers: int = 30
    d_model: int = 2816
    n_heads: int = 16
    n_kv_heads: tuple[int, ...] = ()        # per layer (8 SWA / 2 global)
    is_swa: tuple[bool, ...] = ()           # sliding_window_pattern (True = SWA)
    head_dim_swa: int = 256
    head_dim_global: int = 512
    rope_dims_swa: int = 256
    rope_dims_global: int = 512
    rope_base: float = 1e6                  # global layers
    rope_base_swa: float = 1e4
    window: int = 1024                      # n_swa
    n_ff: int = 2112                        # dense (shared-expert) MLP
    n_ff_exp: int = 704
    n_expert: int = 128
    n_expert_used: int = 8
    vocab_size: int = 262144
    eps: float = 1e-6
    final_logit_softcap: float = 30.0
    attn_scale: float = 1.0                 # gemma4: no pre-attn scaling (qk-norm)
    canvas_length: int = 256
    mask_token_id: int = 4
    bos_token_id: int = 2

    def head_dim(self, il: int) -> int:
        return self.head_dim_swa if self.is_swa[il] else self.head_dim_global

    def rope_params(self, il: int) -> tuple[float, bool]:
        """(freq_base, uses_freq_factors) for layer il."""
        if self.is_swa[il]:
            return self.rope_base_swa, False
        return self.rope_base, True


def config_from_gguf(meta: dict) -> DiffusionGemmaConfig:
    p = "diffusion-gemma."
    c = DiffusionGemmaConfig(
        n_layers=int(meta[p + "block_count"]),
        d_model=int(meta[p + "embedding_length"]),
        n_heads=int(meta[p + "attention.head_count"]),
        n_kv_heads=tuple(int(v) for v in meta[p + "attention.head_count_kv"]),
        is_swa=tuple(bool(v) for v in meta[p + "attention.sliding_window_pattern"]),
        head_dim_swa=int(meta[p + "attention.key_length_swa"]),
        head_dim_global=int(meta[p + "attention.key_length"]),
        rope_dims_swa=int(meta[p + "rope.dimension_count_swa"]),
        rope_dims_global=int(meta[p + "rope.dimension_count"]),
        rope_base=float(meta[p + "rope.freq_base"]),
        rope_base_swa=float(meta[p + "rope.freq_base_swa"]),
        window=int(meta[p + "attention.sliding_window"]),
        n_ff=int(meta[p + "feed_forward_length"]),
        n_ff_exp=int(meta[p + "expert_feed_forward_length"]),
        n_expert=int(meta[p + "expert_count"]),
        n_expert_used=int(meta[p + "expert_used_count"]),
        vocab_size=len(meta["tokenizer.ggml.tokens"]),
        eps=float(meta[p + "attention.layer_norm_rms_epsilon"]),
        final_logit_softcap=float(meta[p + "final_logit_softcapping"]),
        canvas_length=int(meta["diffusion.canvas_length"]),
        mask_token_id=int(meta["tokenizer.ggml.mask_token_id"]),
        bos_token_id=int(meta["tokenizer.ggml.bos_token_id"]),
    )
    assert meta[p + "attention.causal"] is False
    assert int(meta[p + "attention.value_length"]) == c.head_dim_global
    assert int(meta[p + "attention.value_length_swa"]) == c.head_dim_swa
    return c
