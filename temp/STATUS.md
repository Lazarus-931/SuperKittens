# DeepSeek-V2-Lite decode profile — DIAGNOSIS (read-only, no code changes)

Date: 2026-06-08. Host: derek (M4 base, 16 GB). Build: ~/ds_fix (coherent
`dev-sk-deepseek-coherence` dylib + runtime-compiled metallib). GGUF:
DeepSeek-V2-Lite.Q4_K_M.gguf (10.36 GB on disk).

## TL;DR — the cause is NOT compute, it is that the model does not fit in 16 GB

Decode is ~0.55 tok/s (≈1.8 s/tok) instead of the ~25-40 tok/s the bandwidth
roofline predicts. The gap is ~50-220×. It is entirely **swap / VM-compressor
thrash**, not kernel inefficiency, not "all 64 experts", not per-layer CB sync.

A `poem` run (PID 22625, prompt "The capital of France is", max_new 28) sat
**19m50s in process STATE `U`/`stuck` (uninterruptible disk wait) and emitted
ZERO decode tokens.** %CPU 3-6%. It is blocked on pageins/swapins, not running
GPU kernels.

### Live evidence during decode
- `vm.swapusage: used = 12953 MB / free = 359 MB` — swap file essentially FULL.
- Swapins 94.1M, Swapouts 97.9M; Compressions 311M, Decompressions 292M.
- Pages stored in compressor 1.53M × 16KB ≈ **24 GB compressed**.
- Disk lifetime 2019 GB read / 1476 GB written — SSD thrashing.
- RSS after load = **11604 MB** (33.5 s load), then process effectively pinned.

## Resident footprint accounting (why it overflows 16 GB)

Computed from the launcher `alloc_zero` calls + V2-Lite dims (L=27, E=64,
top_k=6, d=2048, moe_int=1408, shared=2816, dense=10944, vocab=102400, nh=16):

| component                         | size      | note |
|-----------------------------------|-----------|------|
| **routed MoE experts**            | **10.90 GB** | gate/up Q4_K, **down forced to Q8_0** |
| shared experts (2, all 27 L)      | 0.93 GB   | fp16 |
| attn weights                      | 0.74 GB   | fp16 |
| embed                             | 0.42 GB   | **full fp16** |
| lm_head                           | 0.42 GB   | **full fp16, UNTIED separate alloc** |
| dense L0 MLP                      | 0.13 GB   | fp16 |
| router + bias + KV + fp32 scratch | <0.2 GB   | |
| **TOTAL weights**                 | **~13.6 GB** | |

On a 16 GB Mac with ~3-4 GB taken by OS/wired pages, ~13.6 GB of weights cannot
be GPU-resident. The OS pushes the cold pages into the compressor and swap. Every
decode token re-touches MoE expert slabs that were evicted → decompress + pagein
on the critical path → ~1.8 s/tok.

### The single decision that breaks the camel's back: down → Q8_0
`weights.c++` (~L607) **deliberately re-quantizes** `ffn_down_exps` from its
native Q4_K_M dtype (Q6_K / Q5_0 / Q4_K, smaller) **UP to a uniform Q8_0**
purely "so the q8_0 per-expert matvec sees one layout." Cost:

- down as Q8_0  = 34 B / 32 weights → **5.30 GB** across 27 layers
- down as Q4_K  = 144 B / 256 weights → **2.55 GB**
- **wasted: ~2.75 GB** — exactly the amount that pushes resident over the swap cliff.

The GGUF on disk is 10.36 GB; we inflate it to ~13.6 GB resident. Undo that
inflation and the model fits with headroom and decode should jump to roofline.

## What is NOT the problem (suspects ruled out)
- **(a) "computes all 64 experts"** — FALSE. `dispatch_layer` MoE dispatches
  `dispatchThreadgroups(z = T*top_k)` and gathers via `moe_top_idx` (top-6 only).
  Per-token MoE bytes read = **0.98 GB** (top-6 × 26 MoE layers) — ≈8 ms @120 GB/s
  if resident. Correct gathered path.
- **(e) per-layer command-buffer sync** — FALSE. `sk_deepseek_forward` builds ONE
  command buffer for the whole 27-layer forward, `commit()` + `waitUntilCompleted()`
  once per token. No per-layer sync.
- **(d) MLA decode inefficiency** — not the driver; attn weights are 0.74 GB total.
- **(f) LM head over 102400 vocab** — a 0.42 GB read/tok (3.5 ms @120 GB/s) plus a
  2-pass argmax; real but second-order vs the 1.8 s swap stall.

The per-token *useful* bandwidth (MoE 0.98 GB + attn + shared + lm_head + embed
≈ 2.0-2.5 GB/tok) → ~17-25 ms/tok @120 GB/s → ~40-60 tok/s ceiling. We get
0.55. The ~50-220× shortfall is 100% paging.

## Optimization plan (ranked by ROI)

### 1. [HIGHEST ROI] Stop inflating down_exps to Q8_0; keep it Q4_K (or native)
- Add a `deepseek_mul_mv_id` path that consumes the native down dtype, or
  re-quantize down to **Q4_K** instead of Q8_0 (kernel already exists:
  `deepseek_mul_mv_id_q4_K`). Saves **~2.75 GB** → resident ~10.8 GB → FITS in
  16 GB with headroom. Expected: from ~0.55 tok/s to roofline ~20-40 tok/s
  (≈40-70×) the moment swap stops. This alone is the fix.
- Risk: numerics. Q4_K down vs Q8_0 down — validate argmax-coherence (the
  coherence build is the baseline). If Q4_K down degrades quality, fall back to
  the native per-layer dtype (Q6_K/Q5_0) rather than uniform Q8_0, which is
  still smaller than Q8_0 and preserves Q4_K_M's intended precision.

### 2. Tie / quantize the LM head + embed (save ~0.4-0.6 GB)
- `w_lm_head` is allocated as a **separate full-fp16 buffer** even though V2-Lite
  may tie embeddings; and embed itself is full fp16 (0.42 GB). Route lm_head via
  a quantized matvec (the q6k/q4k head trick already memo'd for qwen) and/or tie
  to embed. Secondary once #1 lands.

### 3. fp32 MoE activation scratch (small, but free)
- `moe_x_f32 / gate / up / mid / down` are fp32. At T=1 it is ~0.4 MB so not a
  footprint issue, but the fp32 round-trip doubles the activation bandwidth the
  per-expert matvecs read/write. Low priority; revisit after #1.

### 4. Verify fit headroom, then re-measure per-kernel GPU time
- Once #1 makes it resident, re-profile with `GPUEndTime-GPUStartTime` per
  encoder to confirm the breakdown (expected: MoE matvecs dominate ~50-70%,
  lm_head ~10-15%, MLA attn ~10-20%). Only then chase kernel-level wins; today
  any per-kernel timing is meaningless because the GPU is starved waiting on
  pageins.

## Profiling artifacts in this branch
- `temp/profile_mem.sh` — vm_stat/swapusage/process-state snapshot probe.
- `temp/STATUS.md` — this file.
No code changes made (diagnosis only). Fix is a follow-up PR implementing #1.
