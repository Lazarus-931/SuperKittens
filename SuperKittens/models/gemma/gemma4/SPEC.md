# Gemma 4 architecture spec (E2B / E4B)

Derived from HF `transformers/models/gemma4/{configuration_gemma4.py,modeling_gemma4.py}`.

## E2B real config (per `text_config` of `google/gemma-4-E2B-it`)

| field | value | notes |
|---|---|---|
| n_layers | 35 | |
| d_model (hidden_size) | 1536 | |
| n_heads | 8 | |
| n_kv_heads | 1 | |
| head_dim (sliding) | 256 | |
| global_head_dim (full) | 512 | per-layer-type |
| intermediate_size (n_int) | 6144 | base; 2× for kv-shared layers |
| use_double_wide_mlp | True | layers ≥ (n_layers − num_kv_shared_layers) get n_int × 2 |
| num_kv_shared_layers | 20 | layers L15..L34 share KV from prior |
| hidden_size_per_layer_input (PLE_dim) | 256 | PLE table is `[vocab, n_layers * 256]` |
| vocab_size | 262144 | |
| vocab_size_per_layer_input | 262144 | |
| sliding_window | 512 | for sliding_attention layers |
| final_logit_softcapping | 30.0 | `logits = tanh(logits/cap) * cap` after LM head |
| rms_norm_eps | 1e-6 | |
| rope_theta_global | 1_000_000 | full_attention layers |
| rope_theta_local | 10_000 | sliding_attention layers |
| partial_rotary_factor (full only) | 0.25 | only 25% of head_dim is rotated for full_attention |

## Per-layer features

Layer type comes from `text_config.layer_types[L]` (explicit array). For E2B: L4, L9, L14, L19, L24, L29, L34 are `full_attention`; rest are `sliding_attention`. **Last layer must be `full_attention`.** Default fallback pattern is `((L+1) % 6 == 0)` but real configs override.

1. **Per-layer head_dim**: sliding=256, full=512.
2. **Per-layer n_int**: layer L gets `intermediate_size × (2 if L >= n_layers - num_kv_shared_layers and use_double_wide_mlp else 1)`.
3. **KV sharing**: layers ≥ `first_kv_shared_layer_idx = n_layers - num_kv_shared_layers` (E2B: 15). These layers have NO `k_proj`, `v_proj`, `k_norm`, `v_norm` weights. At runtime they reuse K/V from `store_full_length_kv` layer — the last non-shared layer of the same `layer_type` before the boundary.
4. **V-norm**: Q and K get full rmsnorm with learned gamma. V gets rmsnorm WITHOUT gamma (`Gemma4RMSNorm(with_scale=False)`).
5. **Partial RoPE** for full_attention only: only the first `int(head_dim * 0.25)` dims of Q/K are rotated; remaining stay as-is.
6. **PLE injection** per layer: `per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm`, see HF `Gemma4TextDecoderLayer.forward`.

## Final logit softcap

After LM head GEMM produces logits of shape `(batch, seq, vocab)`:
```
logits = tanh(logits / cap) * cap     # cap = 30.0 for E2B/E4B
```

## SK port status (as of 2026-05-12)

✅ Kernels: attn (mha_causal d=128, plus gemma4-specific d=256 local / d=512 global variants in `models/gemma/gemma4/attn.metal`), rope, rmsnorm, gemm, gated_mlp.
✅ Loader infra: `WeightStore`, safetensor reader, BF16→FP16 cast.
✅ Top-level: `sk.load("gemma4-e2b")`, registry.
✅ Config reader: reads `text_config.head_dim`, `global_head_dim`, `hidden_size_per_layer_input` from `config.json`.

❌ **Per-layer n_int** (double-wide MLP): SK assumes uniform `n_int` across layers. Needs per-layer offsets in launcher allocation and `dispatch_layer`.
❌ **KV sharing** (`num_kv_shared_layers`): SK expects every layer to load k/v/k_norm/v_norm tensors. For L≥15 these tensors don't exist in the checkpoint.
❌ **V-norm**: not applied. Add a no-gamma rmsnorm step on V before attention.
❌ **Partial RoPE**: full-attention layers in SK currently rotate the whole head; need to rotate only `partial_rotary_factor * head_dim` dims.
❌ **Final logit softcap**: not applied. Trivial new kernel or fused into argmax.
❌ **AltUp**: skipped intentionally — single-stream simplification. Acceptable for v0.
❌ **Layer types array**: SK computes via `local_period`; should read `text_config.layer_types[]` directly.
❌ **HF comparison harness**: no per-layer diff against reference.

## Recommended porting order

1. HF reference activation dump (one-shot, lets every later step be verified).
2. Layer_types reader + per-layer n_int + KV sharing (together — they all reshape the weight loader).
3. V-norm + partial RoPE + softcap (kernel additions).
4. PLE pipeline validation (already wired structurally).
5. End-to-end logit comparison vs HF.
