"""inference/registry.py — single source of truth for per-model knowledge.

Each row in ``SPECS`` describes everything a family-specific adapter needs to
turn a model name (e.g. ``"qwen3-0.6b"``) into a working ``Model``: HF repo,
on-disk weight directory, optional GGUF artifact, default quant, tokenizer
family, and hardcoded dimensional fallbacks for when ``config.json`` is
missing.

Adding a new model means adding ONE row to this table.

The narrow per-family seam is :pymeth:`from_spec` on each adapter class. The
adapter receives a :class:`ModelSpec` plus runtime overrides (``seq_max``,
``cache_max``, etc.) and is responsible for constructing the model, loading
weights, baking RoPE tables, and attaching a tokenizer.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    family: str                       # "qwen3" | "gemma4" | "mamba2" | "deepseek"
    adapter: str                      # "pkg.mod:ClassName"
    hf_repo: str                      # canonical HuggingFace repo
    weight_dir: str                   # subdir under SuperKittens/model_weights/
    gguf_name: str | None = None      # filename inside weight_dir if GGUF is canonical
    default_quant: str | None = None  # "q8_0" | None
    tokenizer_family: str | None = "qwen3"  # routes through Tokenizer._FAMILY_SPECIALS
    config_path: str | None = None    # subkey in config.json (e.g. "text_config" for gemma4)
    dims: dict[str, Any] = field(default_factory=dict)  # hardcoded fallback dims


SPECS: dict[str, ModelSpec] = {
    "qwen3-0.6b": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-0.6B",
        weight_dir="Qwen3-0.6B",
        gguf_name="Qwen3-0.6B-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="qwen3",
        dims=dict(n_layers=28, d_model=1024, n_heads=16, n_kv_heads=8,
                  head_dim=128, n_int=3072, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=1),
    ),
    "gemma4-e2b": ModelSpec(
        family="gemma4",
        adapter="SuperKittens.models.gemma.gemma4.gemma4:Gemma4",
        hf_repo="google/gemma-4-E2B-it",
        weight_dir="gemma-4-E2B-it",
        default_quant=None,
        tokenizer_family="gemma4",
        config_path="text_config",
        dims=dict(variant="e2b"),
    ),
    "gemma4-e4b": ModelSpec(
        family="gemma4",
        adapter="SuperKittens.models.gemma.gemma4.gemma4:Gemma4",
        hf_repo="google/gemma-4-E4B-it",
        weight_dir="gemma-4-E4B-it",
        default_quant=None,
        tokenizer_family="gemma4",
        config_path="text_config",
        dims=dict(variant="e4b"),
    ),
    # gemma-4-12B-it is model_type "gemma4_unified" (arch
    # Gemma4UnifiedForConditionalGeneration) — structurally distinct from the
    # E2B/E4B/31B "gemma4" arch (NO PLE, NO KV-sharing, full RoPE on global
    # layers, a standalone per-layer output scale, 8-local/1-global KV heads).
    # It gets its OWN adapter (Gemma4Unified) reusing the shared gemma4 launcher +
    # kernels; the E-variant Gemma4 adapter is untouched. Fit-16GB artifact is a
    # Q4_K_M GGUF (HF bf16 safetensors are 23.9 GB). See temp/gemma4-unified/STATUS.md.
    "gemma4-12b-unified": ModelSpec(
        family="gemma4_unified",
        adapter="SuperKittens.models.gemma.gemma4_unified.gemma4_unified:Gemma4Unified",
        hf_repo="google/gemma-4-12B-it",
        weight_dir="gemma-4-12B-it-GGUF",
        gguf_name="gemma-4-12B-it-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="gemma4",
        config_path="text_config",
        dims=dict(variant="12b-unified"),
    ),
    "gemma4-31b": ModelSpec(
        family="gemma4",
        adapter="SuperKittens.models.gemma.gemma4.gemma4:Gemma4",
        hf_repo="google/gemma-4-31B-it",
        weight_dir="gemma-4-31B-it",
        default_quant=None,
        tokenizer_family="gemma4",
        config_path="text_config",
        dims=dict(variant="31b"),
    ),
    "mamba2-130m": ModelSpec(
        family="mamba2",
        adapter="SuperKittens.models.ssm.mamba2.mamba2:Mamba2Model",
        hf_repo="AntonV/mamba2-130m-hf",
        weight_dir="mamba2-130m-hf",
        default_quant=None,
        tokenizer_family=None,  # pre-instruct: no chat template, no specials needed
        dims=dict(n_layers=24, d_model=768, intermediate=1536, n_heads=24,
                  head_dim=64, state_size=128, n_groups=1, conv_kernel=4,
                  chunk_size=256, vocab_size=50288, rms_eps=1e-5,
                  tie_word_embeddings=1),
    ),
    # DeepSeek-V2-Lite: 27 layers, 16 heads, 64 routed + 2 shared experts, top-6.
    # No Q-LoRA (q_proj is direct dense), no group routing, first_k_dense_replace=1.
    # See SuperKittens/models/deepseek/SPEC.md for the full architecture spec and
    # the list of SPEC items still in flight (RoPE interleave, untied LM head,
    # MoE router sigmoid/bias, etc.).
    "deepseek-v2-lite": ModelSpec(
        family="deepseek",
        adapter="SuperKittens.models.deepseek.deepseek:DeepSeek",
        hf_repo="deepseek-ai/DeepSeek-V2-Lite",
        weight_dir="DeepSeek-V2-Lite",
        default_quant=None,
        tokenizer_family="deepseek",
        dims=dict(
            n_layers=27, d_model=2048, n_heads=16,
            qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128,
            q_lora_rank=0,           # V2-Lite: no Q-LoRA
            kv_lora_rank=512,
            n_int=1408,              # moe_intermediate_size
            shared_n_int=2816,       # n_shared_experts (2) * moe_intermediate_size (1408)
            n_expert=64, top_k=6,
            vocab_size=102400,
            eps=1e-6,
            rope_freq_base=10000.0,
            rope_n_ctx_orig=4096,
            rope_scaling_factor=40.0,
            mscale_all_dim=0.707,    # V2-Lite YaRN mscale_all_dim (config.json: 0.707)
            has_q_lora=0,
            router_has_bias=0,
            rope_interleave=1,
            norm_topk_prob=0,
            n_group=0,
            topk_group=0,
            routed_scaling_factor=1.0,
            first_k_dense_replace=1,
        ),
    ),
    # Llama-3.1-Nemotron-Nano-8B-v1: model_type "llama" (LlamaForCausalLM) — the
    # Qwen3 dense decoder minus per-head Q/K-norm, with Llama-3.1 RoPE (theta
    # 500000 + "llama3" frequency scaling). Reuses the qwen launcher/kernels; the
    # Nemotron adapter sets use_qk_norm=0 and re-bakes RoPE with llama3 scaling.
    # Q4_K_M (4.92 GB) fits a 16 GB mini. Untied LM head (output.weight).
    "nemotron-nano-8b": ModelSpec(
        family="nemotron",
        adapter="SuperKittens.models.nemotron.nemotron:Nemotron",
        hf_repo="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        weight_dir="Llama-3.1-Nemotron-Nano-8B-v1-GGUF",
        gguf_name="nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="nemotron",
        dims=dict(n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8,
                  head_dim=128, n_int=14336, vocab_size=128256,
                  eps=1e-5, rope_freq_base=500000.0, tie_word_embeddings=0,
                  use_qk_norm=0,
                  rope_scaling=dict(rope_type="llama3", factor=8.0,
                                    low_freq_factor=1.0, high_freq_factor=4.0,
                                    original_max_position_embeddings=8192)),
    ),
    # Mistral-7B-Instruct-v0.3: model_type "mistral" (MistralForCausalLM) — a
    # Llama-style dense decoder. Reuses the qwen launcher/kernels; the Mistral
    # adapter sets use_qk_norm=0 and rope_interleaved=1 (Llama GGUF NORM rope,
    # theta=1e6, no llama3 rescale). Untied LM head (output.weight). vocab 32768.
    # Q4_K_M (~4.4 GB) fits a 16 GB mini.
    "mistral-7b-v0.3": ModelSpec(
        family="mistral",
        adapter="SuperKittens.models.mistral.mistral:Mistral",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        weight_dir="Mistral-7B-Instruct-v0.3-GGUF",
        gguf_name="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="mistral",
        dims=dict(n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8,
                  head_dim=128, n_int=14336, vocab_size=32768,
                  eps=1e-5, rope_freq_base=1_000_000.0, tie_word_embeddings=0,
                  use_qk_norm=0),
    ),
    # Llama-3.2-3B-Instruct: model_type "llama" (LlamaForCausalLM) — a Llama-arch
    # dense decoder. Reuses the qwen launcher/kernels; the Llama32 adapter sets
    # use_qk_norm=0, rope_interleaved=1 (Llama GGUF NORM rope), and re-bakes RoPE
    # with the llama3 scaling (theta 500000, factor 32). UNLIKE Nemotron/Mistral,
    # the LM head is TIED (tie_word_embeddings=1): the shared core reads
    # token_embd.weight as the head, no untied output.weight. vocab 128256.
    # Q4_K_M (~2.0 GB; embed+output are Q8_0) fits a disk-tight host.
    "llama-3.2-3b": ModelSpec(
        family="llama32",
        adapter="SuperKittens.models.llama32.llama32:Llama32",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        weight_dir="Llama-3.2-3B-Instruct-GGUF",
        gguf_name="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="llama",
        dims=dict(n_layers=28, d_model=3072, n_heads=24, n_kv_heads=8,
                  head_dim=128, n_int=8192, vocab_size=128256,
                  eps=1e-5, rope_freq_base=500000.0, tie_word_embeddings=1,
                  use_qk_norm=0,
                  rope_scaling=dict(rope_type="llama3", factor=32.0,
                                    low_freq_factor=1.0, high_freq_factor=4.0,
                                    original_max_position_embeddings=8192)),
    ),
    # Qwen2.5-3B-Instruct: model_type "qwen2" (Qwen2ForCausalLM) — the Qwen3 dense
    # decoder MINUS per-head Q/K RMSNorm (use_qk_norm=0), same NeoX RoPE (GGML
    # type 2, rope_interleaved=0, theta=1e6) and same gpt2/BPE tokenizer
    # (tokenizer_family="qwen3"). Small sizes (<=3B) tie the LM head
    # (tie_word_embeddings=1; the GGUF has no output.weight). NOTE: Qwen2/Qwen2.5
    # ALSO carry q/k/v projection BIASES (attn_{q,k,v}.bias), which the shared
    # dense launcher does NOT apply — so this row alone is NOT fully config-only
    # until a QKV-bias add lands. attn_qkv_bias is set as a forward-looking flag.
    "qwen2.5-3b": ModelSpec(
        family="qwen2",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen2.5-3B-Instruct",
        weight_dir="Qwen2.5-3B-Instruct-GGUF",
        gguf_name="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="qwen3",
        dims=dict(n_layers=36, d_model=2048, n_heads=16, n_kv_heads=2,
                  head_dim=128, n_int=11008, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=1,
                  use_qk_norm=0, rope_interleaved=0, attn_qkv_bias=1),
    ),
    # Qwen2.5-1.5B-Instruct: same "qwen2" arch as the 3B row above (config-only
    # delta — different dims). Tied LM head (GGUF has token_embd.weight only, no
    # output.weight) and the same Qwen2/2.5 q/k/v projection bias the in-tree
    # bias_add path now applies (attn_qkv_bias=1). Dims verified against the GGUF
    # metadata: block_count=28, embedding_length=1536, head_count=12,
    # head_count_kv=2, head_dim=1536/12=128, feed_forward_length=8960,
    # vocab=151936, rms_eps=1e-6, rope.freq_base=1e6.
    "qwen2.5-1.5b": ModelSpec(
        family="qwen2",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
        weight_dir="Qwen2.5-1.5B-Instruct-GGUF",
        gguf_name="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="qwen3",
        dims=dict(n_layers=28, d_model=1536, n_heads=12, n_kv_heads=2,
                  head_dim=128, n_int=8960, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=1,
                  use_qk_norm=0, rope_interleaved=0, attn_qkv_bias=1),
    ),
    # Yi-1.5-6B-Chat (01.AI): model_type "llama" (LlamaForCausalLM) — a Llama-style
    # dense decoder from a distinct vendor. Reuses the qwen launcher/kernels; the Yi
    # adapter sets use_qk_norm=0 and rope_interleaved=1 (Llama GGUF NORM rope,
    # theta=5e6, no llama3 rescale). Untied LM head (output.weight, Q6_K). vocab 64000.
    # attention_bias=False (no QKV bias). Chat is ChatML (<|im_start|>/<|im_end|>).
    # Q4_K_M (~3.67 GB) fits a 16 GB mini.
    "yi-1.5-6b-chat": ModelSpec(
        family="yi",
        adapter="SuperKittens.models.yi.yi:Yi",
        hf_repo="01-ai/Yi-1.5-6B-Chat",
        weight_dir="Yi-1.5-6B-Chat-GGUF",
        gguf_name="Yi-1.5-6B-Chat-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="yi",
        dims=dict(n_layers=32, d_model=4096, n_heads=32, n_kv_heads=4,
                  head_dim=128, n_int=11008, vocab_size=64000,
                  eps=1e-6, rope_freq_base=5_000_000.0, tie_word_embeddings=0,
                  use_qk_norm=0),
    ),
    # Phi-4-reasoning (14B): model_type "phi3" (Phi3ForCausalLM) — a dense decoder
    # the shared core drives config-only: use_qk_norm=0, rope_interleaved=0 (phi3
    # GGUFs are NOT q/k-permuted → NeoX/type-2, the core default), plain RoPE
    # theta=500000 (rope_scaling null, partial_rotary_factor 1.0 = full 128-dim),
    # untied LM head, no QKV bias, no sliding window. vocab 100352 (tiktoken-style).
    # ARTIFACT: phi3 GGUFs fuse attn_qkv + gate_up(ffn_up); gguf_name points at the
    # one-time repack (models/phi4/repack_phi3_gguf.py — bit-exact row split).
    # Q4_K_M (~8.4 GiB) fits a 16 GB mini with clamped cache_max.
    "phi4-reasoning": ModelSpec(
        family="phi4",
        adapter="SuperKittens.models.phi4.phi4:Phi4",
        hf_repo="microsoft/Phi-4-reasoning",
        weight_dir="Phi-4-reasoning-GGUF",
        gguf_name="microsoft_Phi-4-reasoning-Q4_K_M-sk.gguf",
        default_quant="q4_k_m",
        tokenizer_family="phi4",
        dims=dict(n_layers=40, d_model=5120, n_heads=40, n_kv_heads=10,
                  head_dim=128, n_int=17920, vocab_size=100352,
                  eps=1e-5, rope_freq_base=500000.0, tie_word_embeddings=0,
                  use_qk_norm=0, rope_interleaved=0),
    ),
    # Llama-3.2-1B-Instruct: same Llama arch as 3B but head_dim=64 (the 3B and
    # all other dense families are head_dim=128). Exercises the head_dim-templated
    # decode/causal attention (mha_*_64). Tied LM head, llama3 RoPE scaling.
    # Q4_K_M is ~0.8 GB (embed+output Q8_0).
    "llama-3.2-1b": ModelSpec(
        family="llama32",
        adapter="SuperKittens.models.llama32.llama32:Llama32",
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        weight_dir="Llama-3.2-1B-Instruct-GGUF",
        gguf_name="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="llama",
        dims=dict(n_layers=16, d_model=2048, n_heads=32, n_kv_heads=8,
                  head_dim=64, n_int=8192, vocab_size=128256,
                  eps=1e-5, rope_freq_base=500000.0, tie_word_embeddings=1,
                  use_qk_norm=0,
                  rope_scaling=dict(rope_type="llama3", factor=32.0,
                                    low_freq_factor=1.0, high_freq_factor=4.0,
                                    original_max_position_embeddings=8192)),

    ),
    "qwen3-1.7b": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-1.7B",
        weight_dir="Qwen3-1.7B-GGUF",
        gguf_name="Qwen3-1.7B-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="qwen3",
        dims=dict(n_layers=28, d_model=2048, n_heads=16, n_kv_heads=8,
                  head_dim=128, n_int=6144, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=1),
    ),
    "qwen3-4b": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-4B",
        weight_dir="Qwen3-4B-GGUF",
        gguf_name="Qwen3-4B-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="qwen3",
        dims=dict(n_layers=36, d_model=2560, n_heads=32, n_kv_heads=8,
                  head_dim=128, n_int=9728, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=1),
    ),
    "qwen3-4b-q4km": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-4B",
        weight_dir="Qwen3-4B-GGUF",
        gguf_name="Qwen3-4B-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="qwen3",
        dims=dict(n_layers=36, d_model=2560, n_heads=32, n_kv_heads=8,
                  head_dim=128, n_int=9728, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=1),
    ),
    "qwen3-8b": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-8B",
        weight_dir="Qwen3-8B-GGUF",
        gguf_name="Qwen3-8B-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="qwen3",
        dims=dict(n_layers=36, d_model=4096, n_heads=32, n_kv_heads=8,
                  head_dim=128, n_int=12288, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=0),
    ),
    "qwen3-8b-q4km": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-8B",
        weight_dir="Qwen3-8B-GGUF",
        gguf_name="Qwen3-8B-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="qwen3",
        dims=dict(n_layers=36, d_model=4096, n_heads=32, n_kv_heads=8,
                  head_dim=128, n_int=12288, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=0),
    ),
    "qwen3-14b": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-14B",
        weight_dir="Qwen3-14B-GGUF",
        gguf_name="Qwen3-14B-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="qwen3",
        dims=dict(n_layers=40, d_model=5120, n_heads=40, n_kv_heads=8,
                  head_dim=128, n_int=17408, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=0),
    ),
    "qwen3-14b-q4km": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-14B",
        weight_dir="Qwen3-14B-GGUF",
        gguf_name="Qwen3-14B-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="qwen3",
        dims=dict(n_layers=40, d_model=5120, n_heads=40, n_kv_heads=8,
                  head_dim=128, n_int=17408, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=0),
    ),
    "qwen3-32b": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-32B",
        weight_dir="Qwen3-32B-GGUF",
        gguf_name="Qwen3-32B-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="qwen3",
        dims=dict(n_layers=64, d_model=5120, n_heads=64, n_kv_heads=8,
                  head_dim=128, n_int=25600, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=0),
    ),
    # Granite-4.0-H-1B: IBM granitehybrid — 40 layers, 36 mamba2 + 4 attention
    # (layers 5/15/25/35; per-layer type from GGUF head_count_kv), dense SwiGLU
    # FFN on EVERY layer. Attention is NoPE (no positional encoding) with
    # attention_multiplier 1/128 replacing 1/sqrt(head_dim); embeddings/residuals/
    # logits carry granite scalar multipliers (12 / 0.22 / 1/6). head_dim=128 is
    # why h-1b and not h-micro (h-micro is head_dim=64; SK dense attention is
    # D=128). Tied Q8_0 head. Reuses mamba2 family kernels + shared dense kernels.
    "granite-4.0-h-1b": ModelSpec(
        family="granite",
        adapter="SuperKittens.models.granite.granite:Granite",
        hf_repo="ibm-granite/granite-4.0-h-1b",
        weight_dir="granite-4.0-h-1b-GGUF",
        gguf_name="granite-4.0-h-1b-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="granite",
        dims=dict(n_layers=40, d_model=1536, n_heads=12, n_kv_heads=4,
                  head_dim=128, n_int=4096, d_inner=3072, ssm_n_heads=48,
                  ssm_head_dim=64, ssm_state=128, ssm_n_groups=1, ssm_conv=4,
                  vocab_size=100352, eps=1e-5,
                  embedding_scale=12.0, residual_scale=0.22,
                  attention_scale=0.0078125, logit_scale=6.0),
    ),
    # DiffusionGemma 26B-A4B: block text-diffusion MoE on a gemma4 backbone
    # (llama.cpp PR #24423 is the runtime reference). Stage-1 adapter exposes
    # the unified zero-SC forward only; the entropy-bound sampler is Stage 2.
    # Dims live in the GGUF metadata (config_from_gguf), not here.
    "diffgemma-26b": ModelSpec(
        family="diffgemma",
        adapter="SuperKittens.models.gemma.diffusion.adapter:DiffusionGemma",
        hf_repo="unsloth/diffusiongemma-26B-A4B-it-GGUF",
        weight_dir="diffgemma-26b",
        gguf_name="diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        default_quant="q4_k_m",
        tokenizer_family="gemma4",
    ),
}


def get_spec(name: str) -> ModelSpec:
    if name not in SPECS:
        raise ValueError(f"unknown model {name!r}; known: {list(SPECS)}")
    return SPECS[name]


def list_specs() -> list[str]:
    return list(SPECS)


def _import_adapter(path: str):
    mod_path, cls_name = path.split(":")
    return getattr(importlib.import_module(mod_path), cls_name)


def load(name: str, **overrides):
    """Generic loader: lookup spec, import adapter, build model via from_spec."""
    spec = get_spec(name)
    cls = _import_adapter(spec.adapter)
    return cls.from_spec(spec, **overrides)
