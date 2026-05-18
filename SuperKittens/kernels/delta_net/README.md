# delta_net — Gated DeltaNet kernels (scaffold)

Empty scaffold for the linear-attention component used by Qwen3.5 and friends. SK has nothing here yet; this dir reserves the namespace and captures design intent so the bring-up doesn't fight naming.

## What Gated DeltaNet is

Linear attention with a learned gating mechanism and a delta-rule state update. Replaces softmax attention in most layers of Qwen3.5 (8 × (3 × DeltaNet + 1 × GatedAttention)). Closer in spirit to Mamba/SSMs than to flash-attention: maintains a recurrent state, O(L) instead of O(L²).

References:
- Paper: "Gated Delta Networks: Improving Mamba2 with Delta Rule" (NeurIPS 2024)
- Qwen3.5 model card (architecture pattern)
- Existing SK mamba2/mamba3 kernels in `models/ssm/` — same regime (state-recurrent, linear time), useful prior art

## Target shapes (Qwen3.5-4B)

- d_model = 2560
- n_v_heads = 32
- n_qk_heads = 16
- head_dim_v = 80 (≈ 2560/32)
- head_dim_qk = 160 (≈ 2560/16)
- Hybrid: 3 DeltaNet layers per 1 GatedAttention layer, 8 blocks → 24 DeltaNet + 8 GatedAttention layers

## Kernel surface (proposed, NOT yet implemented)

Five kernels mirror the Mamba2 SSD split:

1. `delta_net_qkv_proj` — fused (Q, K, V) projections + L2-norm on Q/K.
2. `delta_net_gate_proj` — separate gate projection (per-head scalar gate `g`).
3. `delta_net_recurrence` — the inner state update. Per step:
   - `state_new = g · state_old + outer(K, V)` (delta rule with gate)
   - `out = state_new @ Q` (read out)
   - State shape: (H, D_qk, D_v). At Qwen3.5-4B: 16 × 160 × 80 = 204800 fp32 per head per batch.
4. `delta_net_out_proj` — fused output proj + residual.
5. `delta_net_chunked_prefill` — chunked-scan variant for prefill (parallel over chunks like Mamba2 SSD).

## Where the pain will be

- State buffer is large (`H × D_qk × D_v × 4B` fp32). For Qwen3.5-4B: 800 KB per token batch. KV-cache analog for the linear-attn path.
- Hybrid scheduling: every 4th layer is regular Gated Attention. The launcher needs to alternate kernel sets — current SK launchers don't.
- Quant path unclear: DeltaNet's state update accumulates over many tokens — int8/int4 may bleed accuracy fast. fp16 state is the safe default.

## Plan

Stage 1 — port the reference recurrence (single-step, fp16, unfused) from the paper's CUDA code to Metal. Validate against a numpy reference on a 32-token sequence.

Stage 2 — chunked-scan prefill (mirror `models/ssm/mamba2/mamba2_ssd_ref.metal` structure).

Stage 3 — fuse QKV projection + recurrence into one dispatch.

Stage 4 — write the hybrid Qwen3.5 launcher that alternates between DeltaNet and GatedAttention layers.

No work happens here until Stage 1 has a green numerical bench.
