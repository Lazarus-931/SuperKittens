# Kernel Index

Each kernel dir under `SuperKittens/kernels/` ships `.metal` source, `.h`/`.c++`
host dispatch, and (where relevant) MLX baselines under `baseline/`.

Canonical perf numbers: `best.md` (repo root).
Entries tagged **[PR #N]** are on an open branch pending review — not yet on `main`.

## kernels/

| Dir | Contents |
|---|---|
| `activation` | GELU, SiLU, ReLU (half4, no barriers) |
| `attn` | `mha_causal` — production GQA attention (TG-staged, Br=4 queries/TG, d=64/128). **[PR #57]** Q8_0-KV variants (`mha_*_q8`, opt-in `SK_KV_Q8`) |
| `flash_attn` | ds4-derived flash attn, dk/dv up to 512 (decode, MLA) |
| `paged_attn` | Paged KV-cache attention |
| `conv` | conv1d, conv2d, conv3d (conv1d_silu lives in `fusion`) |
| `delta_net` | scaffold only (README) — Gated DeltaNet namespace reserved for Qwen3.5; no kernels yet |
| `fusion` | rms_rope, rms_residual, add_rmsnorm, gemm_bias_act, gemm_res_norm, gated_mlp (+ bf16/geglu variants), gemv_bf16_m1, gemv_swiglu_m1, kv_up_pair, silu_mul, conv1d_silu |
| `gemm` | **decode matvecs (M=1):** q8_0 (+ bf16 / addres variants), q4k, q6k, q2k, iq2xxs, q8_0_swiglu_m1; legacy fp16/fp8 dense GEMM + gemv. **[PR #63]** q3k/q5k canonical matvec (fit-enablers). **[PR #55]** `gemm_mma` + `gemm_mma_smallm` — batched seq>1 MMA GEMM (f16/Q8_0/Q4_K, `simdgroup_float8x8`) for prefill/verify |
| `moe` | router, router_v3, down_scatter, swiglu_pair (+ Q2K / IQ2XXS quantized) |
| `ops` | add, cast, causal_mask, kv_cache, sample (argmax_2pass / argmax_bf16 / sample), split, transpose. **[PR #57]** `kv_cache_write_q8` |
| `rotary` | RoPE on Q/K (standalone path; usually fused via `fusion/rms_rope`) |
| `swiglu` | fused_swiglu: `silu(gate) * up` |
| `utils` | rmsnorm, layernorm, embedding |

Auto-dispatch: `attn.h` routes `head_dim==64` → FA64, else MHA.
Decode is M=1 matvec; prefill/verify (seq>1) routes to `gemm_mma` **[PR #55]**.

## Settled findings (M4 base, decode)

- **Decode is GPU-bandwidth-bound.** ~99% GPU-wait; CPU-encode is 0.4–0.7% of wall
  time. ICB / dispatch-fusion / encode-side levers are dead for decode — the win is
  weight bytes moved, not dispatch overhead.
- **Q4_K_M is the decode sweet spot.** Sub-Q4 weight-quant (Q2_K, Q3_K) is a *net loss*:
  the dequant is compute-bound (q3k matvec peaks ~67 GB/s vs q4k ~107 GB/s), so fewer
  bytes don't translate to tok/s. Q3_K/Q5_K are kept only as **fit-enablers** (keep
  projections packed end-to-end so a larger model fits 16 GB) — see **[PR #63]**.
- **Batched GEMM wins prefill, not decode.** `gemm_mma` **[PR #55]** gives 2.5–5.8× TTFT
  and sub-linear seq>1 cost; decode (M=1) is unchanged. Spec-decode still loses on M4 4B
  (draft tax + accept ceiling > the verify saving); 8B under test.
- **Q8_0 KV cache** **[PR #57]** is a memory/context feature (~2× cache_max), not a speed
  win — KV size isn't the decode lever (weight bandwidth is). Default path byte-identical.

## models/

Per-model orchestration (weights loader, launcher C ABI, Python ctypes wrapper,
model-specific kernels):

| Dir | Model family |
|---|---|
| `qwen` | Qwen3 — Q4_K_M end-to-end (4B/8B/14B fit 16 GB), Q6_K LM-head routing. **[PR #64]** prefill logits OOB + last-row index fix; **[PR #55]** seq>1 prefill via `gemm_mma` |
| `gemma` | Gemma 2/3/4 (incl. gemma4 SWA attn at d=256/512) |
| `deepseek` | DeepSeek (MLA via flash_attn dk=dv=512); V2-Lite e2e wired but incoherent (de-interleave / RoPE / T>1 prefill blockers) |
| `ssm/mamba2` | Mamba2 SSD. **[PR #60]** `mamba2_ssd.metal` rewritten HF-correct (= ref, ~2.7× faster); **[PR #62]** 130m e2e coherent (= HF token-for-token). See `ssm/mamba2/STATUS.md` |
| `ssm/mamba3` | Mamba3 SSM (pre_ssm RMSNorm+RoPE, mamba3_ssm, post_ssm gate). **[PR #61]** score-MMA numerics + tail-chunk fix (no hang to L=32768) |
| `load` | Shared HF safetensors loader |
