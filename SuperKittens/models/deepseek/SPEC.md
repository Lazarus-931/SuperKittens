# DeepSeek V3 / V4 Flash architecture spec

Derived from HF `transformers/models/deepseek_v3/{configuration_deepseek_v3.py,modeling_deepseek_v3.py}`
and `transformers/models/deepseek_v2/configuration_deepseek_v2.py`, plus antirez's
plain-C reference at `~/ds4/ds4.c` for V4-Flash-specific GGUF metadata.

There is **no** `deepseek_v4` directory in HF transformers as of 2026-05-13.
V4 Flash = V3 architecture + (sparse-attention indexer, sliding window, FP8 KV
cache, IQ2_XXS / Q2_K routed-expert quant, group-restricted top-k routing
with `expert_group_count` metadata). The HF V3 implementation is authoritative
for the dense math; V4-only features are flagged below.

## V3 real config (defaults from `DeepseekV3Config`, configuration_deepseek_v3.py:71-104)

| field | value | notes |
|---|---|---|
| vocab_size | 129280 | |
| hidden_size (d_model) | 7168 | |
| intermediate_size | 18432 | dense MLP (used only in `first_k_dense_replace` shallow layers) |
| moe_intermediate_size | 2048 | per-expert MLP width |
| num_hidden_layers | 61 | (HF marks `model.layers.61.*` as ignored on load, modeling_deepseek_v3.py:547 — actual depth is 61 used + 1 MTP slot) |
| num_attention_heads | 128 | |
| num_key_value_heads | 128 | MLA, not GQA — every head has its own decompressed K/V |
| n_shared_experts | 1 | |
| n_routed_experts | 256 | |
| routed_scaling_factor | 2.5 | scales top-k weights, modeling_deepseek_v3.py:236 |
| kv_lora_rank | 512 | |
| q_lora_rank | 1536 | (V3 has it; V2-Lite has q_lora_rank=None → no LoRA on Q) |
| qk_rope_head_dim | 64 | |
| qk_nope_head_dim | 128 | |
| qk_head_dim | 192 | = nope+rope, set in `__post_init__` (configuration_deepseek_v3.py:110) |
| v_head_dim | 128 | **V dim != QK dim** |
| n_group | 8 | groups of routed experts |
| topk_group | 4 | groups kept |
| num_experts_per_tok (top_k) | 8 | |
| first_k_dense_replace | 3 | layers 0..2 use plain MLP, layers 3..60 use MoE |
| norm_topk_prob | True | divides top-k weights by sum |
| hidden_act | silu | |
| max_position_embeddings | 4096 | base; YaRN extends |
| rms_norm_eps | 1e-6 | |
| rope_interleave | True | uses `apply_rotary_pos_emb_interleave` not standard NEOX (modeling_deepseek_v3.py:444) |
| attention_bias | False | |
| tie_word_embeddings | False | lm_head is a separate weight (configuration_deepseek_v3.py:100) |

`rope_parameters` in real released V3: `{"rope_type":"yarn","factor":40.0,"original_max_position_embeddings":4096,"beta_fast":32,"beta_slow":1,"mscale":1.0,"mscale_all_dim":1.0,"rope_theta":10000.0}`.

## V2-Lite config (for the validation dump target)

`deepseek-ai/DeepSeek-V2-Lite`, ~15.7B params, ~31GB bf16. Uses `DeepseekV2Config`
(architecture-equivalent except simpler routing) with notable deltas:
- `q_lora_rank = None` → no Q LoRA; uses direct `q_proj`
- `n_routed_experts = 64`, `n_shared_experts = 2`, `num_experts_per_tok = 6`
- `moe_intermediate_size = 1408`
- `n_group = None`, `topk_group = None`, `topk_method = "greedy"` (no group gating)
- `routed_scaling_factor = 1.0`, `norm_topk_prob = False`
- `first_k_dense_replace = 1`
- `hidden_size = 2048`, `num_hidden_layers = 27`, `num_attention_heads = 16`
- Same MLA dims: `kv_lora_rank=512`, `qk_rope_head_dim=64`, `qk_nope_head_dim=128`, `v_head_dim=128`, `qk_head_dim=192`

V2-Lite shares 100% of the MLA + MoE machinery with V3; only the routing-group
logic and Q-LoRA path differ, both of which are config-gated.

## Architecture summary

1. **Embedding**: `nn.Embedding(vocab_size, hidden_size)`. **No sqrt scaling** —
   modeling_deepseek_v3.py:594-595 just does `embed_tokens(input_ids)`.
2. **Per-layer block** (`DeepseekV3DecoderLayer.forward`, lines 497-526):
   `h = h + attn(rmsnorm(h)); h = h + mlp(rmsnorm(h))`. Plain pre-norm residual,
   no logit gating, no PLE.
3. **MLA attention** (`DeepseekV3Attention.forward`, lines 416-479):
   - Q path: `q_a_proj(d_model→q_lora_rank)` → `q_a_layernorm` (RMSNorm on q_lora_rank)
     → `q_b_proj(q_lora_rank → n_heads*qk_head_dim)` → reshape `(B,H,S,192)` →
     split into `q_pass (128)` and `q_rot (64)`.
   - KV path: `kv_a_proj_with_mqa(d_model → kv_lora_rank+qk_rope_head_dim)` →
     split into `compressed_kv (512)` and `k_rot (64)`. Then
     `kv_a_layernorm` on the 512 part, then `kv_b_proj(512 → H*(128+128))` →
     split into `k_pass (128)` and `value_states (128)`.
   - `k_rot` is broadcast across all heads (single MQA-rope slot,
     modeling_deepseek_v3.py:441, 448).
   - **Partial RoPE**: only the rope-portion (`q_rot`, `k_rot`) gets rotated;
     the nope halves pass through unchanged (lines 444-451). RoPE uses
     `apply_rotary_pos_emb_interleave` because `rope_interleave=True` —
     this reshapes `(d/2, 2)` → transposed pairs before standard rotate_half,
     which is **different from NEOX** (lines 320-355).
   - Attention scale = `qk_head_dim ** -0.5 = 1/sqrt(192)`. With YaRN, scale is
     additionally multiplied by `mscale*mscale` (line 414).
   - V is padded with zeros to qk_head_dim for FA backends and unpadded after
     (lines 456-458, 474-475).
   - `o_proj(H * v_head_dim → d_model)`.
4. **MoE FFN** (`DeepseekV3MoE.forward`, lines 239-247):
   `out = experts(x, topk_idx, topk_w) + shared_experts(x)`. Residual is
   shared+routed; the layer-level residual is added outside.
5. **Router** (`route_tokens_to_experts`, lines 214-237):
   - logits = `sigmoid(x @ W_router.T)` (fp32, line 150)
   - score_for_choice = logits + `e_score_correction_bias` (per-expert bias)
   - Group restriction: reshape `(n_group, experts_per_group)`, take top-2 per
     group, sum → group scores; keep `topk_group` best groups; mask others to
     -inf. Then top-k over allowed experts. **V2-Lite skips this entire block**
     (n_group=None).
   - top-k weights = `logits.gather(topk_indices)` (raw sigmoid, not scores)
   - if `norm_topk_prob`: divide by sum (+1e-20)
   - multiply by `routed_scaling_factor`
6. **Shared experts** use one `DeepseekV3MLP` with `intermediate_size =
   moe_intermediate_size * n_shared_experts` (line 205) — i.e. a single fused
   wide MLP, not n_shared separate experts.
7. **Routed expert MLP**: SwiGLU. Weight `gate_up_proj` is `(E, 2*I, H)`
   packed, split via `.chunk(2)` (lines 163, 185). `act = silu(gate) * up`.
8. **Final RMSNorm**: applied after the last decoder layer (`self.norm`, line 627).
9. **LM head**: plain `nn.Linear(d_model, vocab, bias=False)` (line 644).
   **No softcap**, **no scaling**, weights not tied (configuration_deepseek_v3.py:100).
10. **Dtype**: HF weights ship in bf16; `RMSNorm` upcasts to fp32 internally
    (lines 48-52), as does the router (line 150) and RoPE cos/sin (line 114).
    `e_score_correction_bias` is kept in fp32 (`_keep_in_fp32_modules_strict`,
    line 546).

## V4 Flash deltas (from `~/ds4/ds4.c` metadata keys, lines 1250-1262)

| feature | V3 | V4 Flash |
|---|---|---|
| `attention.sliding_window` | absent | present (mixed local/global per-layer) |
| `attention.indexer.{head_count,key_length,top_k}` | absent | sparse-attention top-k token gate added in front of attention |
| KV cache dtype | bf16/fp16 | FP8 E4M3 with per-head scale (`dsv4_e4m3fn_dequant_cpu` ds4.c:1459, `dsv4_fp8_kv_quantize_row_inplace_cpu` ds4.c:1489) |
| Routed gate/up | bf16 | IQ2_XXS (`block_iq2_xxs`) |
| Routed down | bf16 | Q2_K (`block_q2_K`) |
| Activation dtype for expert dot | fp32 | Q8_K-quantized activations to match int2 weights (ds4.c:1509) |
| MTP slot | layer 61 ignored | layer 61 = multi-token-prediction head |

Everything else (MLA shape, RMSNorm form, RoPE interleave, group routing, shared
experts) is identical between V3 and V4 Flash.

## SK port status table (as of 2026-05-13)

Cross-checked against `SuperKittens/models/deepseek/{deepseek_model.h,weights.c++}`.

| feature | status | notes |
|---|---|---|
| Embedding lookup (no scaling) | ✅ keep | `dispatch_model` does plain gather (deepseek_model.h:638-650) |
| Pre-attn RMSNorm | ✅ keep | (deepseek_model.h:196) |
| Q LoRA path (q_a / q_a_norm / q_b) | ✅ keep | (deepseek_model.h:199-206) — but **V2-Lite has no q_lora**, fix needed if porting Lite |
| KV LoRA + split into c_kv / k_rot | ✅ keep | split_packed kernel handles it (deepseek_model.h:212-223) |
| kv_a_layernorm on c_kv only | ✅ keep | (deepseek_model.h:225) |
| kv_b_proj fused as kv_up_pair | ✅ keep | tile-MMA fuses K/V up-projection (deepseek_model.h:336-353) |
| Partial RoPE on q_rot only (64 of 192) | ✅ keep | rope_tail dispatched with `n_dims=qk_rope_dim` (deepseek_model.h:247, 284) |
| RoPE on shared k_rot, broadcast across heads | ⚠ wrong-form-needs-fix | SK rotates k_pe as `(T, 64)` flat; broadcast across heads at FA time is implicit but verify FA reads K with stride 0 across head dim, otherwise need explicit expand |
| **RoPE interleave (rope_interleave=True)** | ❌ missing | SK calls `rope_tail` with `mode=2` (NEOX). HF uses `apply_rotary_pos_emb_interleave` (modeling_deepseek_v3.py:320, 444), which reshapes `(d/2,2).transpose` before rotate_half — different pair ordering. Need to match exactly or pre-permute weights |
| Attention scaling = 1/sqrt(qk_head_dim=192) | ✅ keep | `a.scale = 1/sqrt(dk)`, dk=192 (deepseek_model.h:394) |
| YaRN mscale² multiplier on scale | ❌ missing | only applied when rope_type != "default" (modeling_deepseek_v3.py:412-414). Real V3 ships with YaRN |
| Causal mask | ✅ keep | (deepseek_model.h:406-415) |
| Flash-attention vec | ✅ keep | (deepseek_model.h:418) |
| O proj | ✅ keep | (deepseek_model.h:439) |
| Residual + pre-MLP RMSNorm fused | ✅ keep | add_rmsnorm (deepseek_model.h:486-499) |
| Shared expert (single fused MLP, width = moe_int*n_shared) | ⚠ wrong-form-needs-fix | SK uses `shared_n_int` param but `dispatch_shared_expert` passes `K_v = p.d_model` to gated_mlp instead of `p.shared_n_int` for the inner width (deepseek_model.h:464). Looks like the GEMM-tile sizing is wrong; verify against gated_mlp kernel contract |
| Router: sigmoid + e_score_correction_bias | ❌ missing | SK MoE router (see kernels/moe/) needs to verify both sigmoid (not softmax) and the additive bias. Check `MoeFfnPSOs.router` impl |
| Group-restricted top-k (n_group=8, topk_group=4) | ❌ missing | SK MoE router does flat top-k; group gating not implemented. V2-Lite path doesn't need it, V3/V4 does |
| `routed_scaling_factor` (2.5 for V3) | ❌ missing | not wired into MoeFfnParams |
| `norm_topk_prob` (True for V3) | ❌ missing | verify in moe kernel |
| Routed expert SwiGLU with gate/up fused (`gate_up_proj`) | ✅ keep | swiglu_pair kernel exists |
| First-k-dense-replace (layers 0..2 use plain MLP, not MoE) | ❌ missing | `dispatch_layer` always routes through `dispatch_moe_ffn` — needs per-layer branch |
| Shared experts + routed sum | ✅ keep | add then MoE residual (deepseek_model.h:503-530) |
| Final RMSNorm | ✅ keep | (deepseek_model.h:741) |
| LM head (untied) | ⚠ wrong-form-needs-fix | `dispatch_model` reuses `W.w_embed` as the LM head (deepseek_model.h:753). V3 has `tie_word_embeddings=False`, needs a separate `w_lm_head` buffer |
| Logit softcap | ✅ keep | **none in V3/V4** — SK correctly omits |
| BF16→FP16 load cast | ✅ keep | per scaffold notes |
| V4 sliding window | ❌ missing | per-layer mask radius needed |
| V4 sparse attention indexer | ❌ missing | top-k key gating before FA |
| V4 FP8 KV cache (E4M3) | ❌ missing | currently fp16 in `c_kv_cache`/`k_pe_cache` |
| V4 IQ2_XXS / Q2_K expert quant | ✅ keep (declared) | `MoeQuant::FP16 / Q4_K / Q2_K / IQ2_XXS` enum exists; verify kernels actually implement IQ2_XXS dot |
| V4 MTP head | ❌ missing | optional for inference; can skip in v0 |

**Tally**: 14 ✅ keep, 4 ⚠ needs-fix, 11 ❌ missing.

## Recommended porting order

1. Run validation harness in `temp/deepseek_validate/` against V2-Lite (no group
   routing, no Q-LoRA — simplest config) to lock down MLA + RMSNorm + RoPE
   interleave + flash-attn numerics.
2. Fix RoPE interleave (or pre-permute weights at load time).
3. Fix LM head to use a separate buffer (`tie_word_embeddings=False`).
4. Fix shared-expert GEMM dimensions.
5. Add `first_k_dense_replace` branch in `dispatch_layer`.
6. Add MoE sigmoid + bias + `routed_scaling_factor` + `norm_topk_prob`.
7. Add group-restricted top-k routing (V3/V4 only).
8. Add YaRN mscale² to attention scale.
9. V4-only: sliding window, sparse indexer, FP8 KV.
