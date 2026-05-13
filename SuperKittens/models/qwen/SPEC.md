# Qwen3 architecture spec

Derived from HF `transformers/models/qwen3/{configuration_qwen3.py,modeling_qwen3.py}`.

## Config (Qwen3-0.6B reference; verify against actual `config.json`)

| field | Qwen3 default | Qwen3-32B |
|---|---|---|
| hidden_size (d_model) | 4096 | 5120 |
| intermediate_size (n_int) | 22016 | 27392 |
| num_hidden_layers | 32 | 64 |
| num_attention_heads | 32 | 64 |
| num_key_value_heads | 32 | 8 (GQA 8:1) |
| head_dim | 128 | 128 |
| vocab_size | 151936 | 151936 |
| max_position_embeddings | 32768 | 32768 |
| rope_theta | 1_000_000 | 1_000_000 |
| rms_norm_eps | 1e-6 | 1e-6 |
| hidden_act | "silu" | "silu" |
| use_sliding_window | False | False |
| sliding_window | None | None |
| tie_word_embeddings | False | False |
| attention_bias | False | False |

## Architecture (Qwen3 — same for all dense variants)

1. **Embedding**: `embed_tokens[input_ids]` — **no scaling** (unlike Gemma 4).
2. **Per layer L** (uniform across all layers):
   - `input_layernorm` (RMSNorm) → x_norm
   - `self_attn`:
     - q = q_proj(x_norm) → reshape to (B, T, n_heads, head_dim)
     - k = k_proj(x_norm) → reshape to (B, T, n_kv_heads, head_dim)
     - v = v_proj(x_norm) → reshape to (B, T, n_kv_heads, head_dim)
     - **q_norm(q)**, **k_norm(k)** — per-head RMSNorm on head_dim
     - apply_rotary_pos_emb(q, k, cos, sin) — standard NeoX-style half-split rotation, full head_dim rotated
     - attention with scaling = 1/√head_dim, causal mask
     - o_proj(attn_out) → residual_out
   - residual = x + residual_out
   - `post_attention_layernorm`(residual) → mlp_in
   - `mlp`: down_proj(silu(gate_proj(mlp_in)) * up_proj(mlp_in))
   - residual = residual + mlp_out
3. **Final**: `model.norm`(residual) → `lm_head` → logits. **No softcap.**

## RMSNorm form (modeling_qwen3.py:49-67)

```python
hidden_states = hidden_states.to(float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * rsqrt(variance + eps)
return weight * hidden_states.to(input_dtype)
```

Standard form: `weight * (x / rms)`. **Initialized to ones** (so storing close-to-1 values is normal). Matches SK's existing `rmsnorm` kernel form.

## HF tensor names

- `model.embed_tokens.weight` (vocab_size, hidden_size)
- `model.norm.weight` (hidden_size,)
- Per layer L:
  - `model.layers.{L}.input_layernorm.weight` (hidden_size,)
  - `model.layers.{L}.post_attention_layernorm.weight` (hidden_size,)
  - `model.layers.{L}.self_attn.q_proj.weight` (n_heads × head_dim, hidden_size)
  - `model.layers.{L}.self_attn.k_proj.weight` (n_kv_heads × head_dim, hidden_size)
  - `model.layers.{L}.self_attn.v_proj.weight` (n_kv_heads × head_dim, hidden_size)
  - `model.layers.{L}.self_attn.o_proj.weight` (hidden_size, n_heads × head_dim)
  - `model.layers.{L}.self_attn.q_norm.weight` (head_dim,)
  - `model.layers.{L}.self_attn.k_norm.weight` (head_dim,)
  - `model.layers.{L}.mlp.gate_proj.weight` (n_int, hidden_size)
  - `model.layers.{L}.mlp.up_proj.weight` (n_int, hidden_size)
  - `model.layers.{L}.mlp.down_proj.weight` (hidden_size, n_int)
- `lm_head.weight` (vocab_size, hidden_size) — only if `tie_word_embeddings=False`

Weights stored as BF16; SK needs BF16→FP16 cast (already implemented in our generic loader path; verify Qwen path uses it).

## SK port status (cross-check)

✅ **Kernels** all needed are already in SK:
  - `mha_causal` (now GQA-aware after the attn ABI fix)
  - `rope_qk` (split-half NeoX-style, matches HF apply_rotary_pos_emb)
  - `rmsnorm` (form matches HF)
  - `gemm_fp16` (with M=1 gemv fast-path)
  - `gated_mlp` / `swiglu`
  - `add_rmsnorm` (fused residual + norm)

✅ **Orchestrator**: `models/qwen/qwen_model.h` has the right per-layer dispatch order.

⚠️  **`weights.c++` (Qwen)**: currently does NOT BF16 cast. Look: `copy_into` uses `memcpy` only. For real Qwen3 weights (BF16), this writes garbage. **Fix needed: mirror Gemma 4's `bf16_to_fp16` helper.**

⚠️  **Embedding scaling**: Qwen3 does NOT scale. Our generic loader doesn't either. Confirmed match.

⚠️  **GQA n_kv_heads**: Qwen3-32B uses 8 KV heads vs 64 Q heads. Our mha_causal handles this via the GQA tile-sharing optimization. Verified ABI matches.

⚠️  **`tie_word_embeddings=False` for 32B**: SK currently uses transB=1 on the embedding table for LM head (tied). For Qwen3-32B we need a separate `lm_head` load + standard GEMM. Smaller models like Qwen3-0.6B / 1.7B have `tie_word_embeddings=True`; OK there.

❌ **Real-checkpoint validation**: never done. Synthetic safetensors smoke passes; semantic correctness against HF unknown.

## Recommended order (mirrors Gemma 4 plan)

1. Add BF16→FP16 cast to `models/qwen/weights.c++` (mirror Gemma 4's helpers).
2. Pick a small Qwen3 variant that fits 16GB Mac: **Qwen3-0.6B** (~1.2GB bf16) or Qwen3-1.7B (~3.5GB).
3. Run HF reference dump in `temp/qwen3_validate/`.
4. Run SK forward + compare last logits.
5. If divergence, drill into per-layer dumps.

## Notes / minor differences from Gemma 4

- No PLE → no `embed_tokens_per_layer`.
- No KV sharing.
- No partial RoPE.
- No logit softcap.
- No alternating local/global.
- No per-layer-variable dims.

Qwen3 is the right validation target: any divergence is a clean SK runtime bug, not an unimplemented architectural feature.
