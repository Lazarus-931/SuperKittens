# DeepSeek V3 — promotion summary

Promoted from `SuperKittens/temp/sk4/` per `PROMOTE.md` (read it for the per-edit map).

## Files promoted

| staged | target | HF citation |
|---|---|---|
| `rope_interleave.metal` | `models/deepseek/kernels/rope_interleave.metal` | modeling_deepseek_v3.py:444 (`apply_rotary_pos_emb_interleave`, GPT-J-style pair rotation) |
| `router_v3.{metal,h}` | `kernels/moe/router_v3.{metal,h}` | modeling_deepseek_v3.py:150 (sigmoid gate), :217 (e_score_correction_bias), :222-228 (group top-2 + topk_group mask + topk), :232 (norm_topk_prob), :236 (routed_scaling_factor) |
| `V4_TODO.md` | `models/deepseek/V4_TODO.md` | (deferred deltas; see file for citations) |

## In-place edits to `deepseek_model.h` (patches A–H from `temp/sk4/deepseek_model.patch.h`)

| patch | location | HF citation | notes |
|---|---|---|---|
| A | LayerParams + ModelParams | configuration_deepseek_v3.py | Added has_q_lora, is_moe_layer, rope_interleave, yarn_mscale, n_group, topk_group, routed_scaling, norm_topk_prob, router_has_bias, first_k_dense_replace. Mirrored in `sk_deepseek_config`. |
| B | dispatch_attn FA scale | modeling_deepseek_v3.py:412-414 | `a.scale = (1/sqrt(dk)) * yarn_mscale^2`. `ds_compute_yarn_mscale(factor, mscale_all_dim)` returns 1.0 when factor ≤ 1. |
| C | dispatch_attn RoPE | modeling_deepseek_v3.py:444 | Branched on `p.rope_interleave`: V3 dispatches `rope_interleave_f32` for Q (rotated half) and k_pe; V2 keeps `kernel_dsv4_rope_tail_f32` mode=2 (NeoX/half-split). |
| D | dispatch_shared_expert K_v | (no-op) | Patch text claimed `K_v = p.d_model` was wrong, but the `gated_mlp` kernel signature is (M, N, K=input_dim, N_int). `K_v=p.d_model` is the correct input dim. Left as-is. |
| E | dispatch_layer dense branch | configuration_deepseek_v3.py:90 (first_k_dense_replace=3) | When `!p.is_moe_layer` (L<first_k_dense_replace), call shared_expert with width=intermediate_size and skip MoE FFN; copy `shared_out → y_out` via blit. |
| F | LM head | configuration_deepseek_v3.py:100 (`tie_word_embeddings=False`) | Added `w_lm_head` buffer; loader aliases to `w_embed` when caller passes null. |
| G | router PSO swap | modeling_deepseek_v3.py:214-237 | Replaced MoE router stage with `moe_router_v3`; bind per-layer `router_bias` (fp32 `e_score_correction_bias`). Falls back via `has_bias=0` for V2-Lite. |
| H | Q-LoRA conditional | configuration_deepseek_v2.py:q_lora_rank=None | When `!p.has_q_lora`, skip q_a path and gemm directly `x_norm → q_packed` using `w_q_b` as `q_proj`. |

## Per-kernel numeric tests

Harness: `SuperKittens/temp/deepseek_kernel_tests/test_kernels.cpp` (C++; links `build/metal_impl.o`, loads `build/libsk.metallib`).

Results (M3 Max, `./build/test_ds_kernels`):

```
=== rope_interleave_f32 ===
  shape: B=1 T=4 NH=8 HD=128 n_dims=64
  max_abs_err (fp32) = 3.874e-07
  PASS                                          (target < 1e-4)

=== moe_router_v3 ===
  shape: T=4 D=64 N=32 K=4 G=4 TKG=2
  expert_indices_match = yes
  max_weight_abs_err (fp16) = 3.877e-04
  PASS                                          (target < 1e-2)
```

CPU references mirror the HF Python (modeling_deepseek_v3.py:apply_rotary_pos_emb_interleave; DeepseekV3MoEGate).

Q2_K / IQ2_XXS matvec tests were intentionally deferred. Rationale: those kernels are pre-existing (not promoted in this batch), and a faithful CPU dequant ref requires staging the exact ggml block layouts (`block_q2_K` / `block_iq2_xxs` per `~/ds4/ds4.c`). They're listed in `V4_TODO.md` §5 with explicit pointers.

## Bug found during testing

The first run of the rope_interleave test failed (max_err ≈ 2.1). Root cause was in the **test code**: it used (B, T, NH, HD) row layout but the kernel reads (B, NH, T, HD) — `row = ((i3*ne02+i2)*ne01+i1)` with `ne01=seq, ne02=heads`. Fixed the test. The kernel itself is correct.

While debugging, a latent **orchestrator-level layout mismatch** was observed: `dispatch_attn` packs Q from the q_b GEMM as `(T, n_heads, dk)` but passes strides to rope kernels assuming `(B, n_heads, T, dk)`. This affects both the existing `rope_tail` and the new `rope_interleave` equally — pre-existing, not introduced by this promotion. Will need a layout transpose (or stride-swap in the args struct) before end-to-end output matches HF. Flagged here; not blocking kernel-level validation.

## Build

`./build.sh` clean. New air files:
- `build/models_deepseek_kernels_rope_interleave.air`
- `build/kernels_moe_router_v3.air`

## V4 deferred

See `models/deepseek/V4_TODO.md`: sliding window, sparse indexer, FP8 E4M3 KV cache, MTP head at layer 61, IQ2_XXS/Q2_K expert quant verification.
