# SuperKittens Deep-Redesign Campaign — PROGRESS

Single source of truth for the autonomous optimization campaign. Relentless mode:
chase improvements, iterate on wins, never stop until the user says stop.

## Operating model
- **Branches:** feature branches OK, push to remote OK. **NEVER touch `main` directly — main is PR-only.** Integration branch: `dev-sk-redesign`. Per-effort branches: `dev-sk-<area>`.
- **Agents:** each deep effort runs worktree-isolated. Build locally (only this Mac has `xcrun metal`), serialized under `lockf -t 600 /tmp/sk_build.lock` (flock absent on macOS; concurrent `xcrun metal` crashes the host). Bench on minis under `lockf -t 1200 /tmp/sk_bench.lock`.
- **Minis (Tailscale):** derek (`dereks-mac-mini` 100.64.169.42) UP; amelia (`amelias-mac-mini` 100.102.119.75) UP; lexie (`github-lexies-mac-mini`) DOWN (ssh timeout). minis are CommandLineTools-only → build .metallib/.dylib locally, copy over.
- **Bench protocol:** prompt "Generate a poem about pizza dough"; 2 warmup + 30s cooldown + 5-rep median; same host/build A/B; report tok/s + coherence + GB/s where relevant. Compare own same-config Q4_K_M baseline (tok/s is cache_max-dependent).
- **Validated win → open a PR to main** (do not merge without orchestrator/user review). Negatives get logged here, not integrated.
- **No Co-Authored-By trailers** (classifier blocks them). WHY-only comments.

## Current decode baselines (Q4_K_M, M4 base, cache_max-dependent)
4B 38.48 · 8B 21.09 · 14B 11.28 tok/s. Decode sits ~71% of M4 roofline; kernels hit 78–95% BW → ~20–25% is inter-dispatch overhead.

## Settled findings (do not re-attempt)
- **Q2_K @4B**: net loss 0.68× + garbage coherence + Q3_K-tensor host-dequant byte-explosion.
- **Q3_K decode @4B/8B/14B**: 0.77×/0.91×/0.62× — q3k matvec is COMPUTE-bound (30–56 GB/s vs q4k 106–110); q5k also (52). Coherence clean; speed regresses. **Q4_K_M is the M4 decode sweet spot.** q3k/q5k kernels kept as FIT enablers only.
- **Spec-decode** regresses (0.54×) without batched GEMM — verify forward re-pays per-token compute.
- **mha_causal chunked-forward fix** merged (PR #54, main 49f9174).
- **Q6_K-LM-head routing** win merged (PR #53): 4B/8B/14B +8.5–15.4%.

## Status legend
`TODO` · `WIP <agentid>` · `BENCH` (validated, PR pending) · `WIN` (PR open/merged) · `NEG` (negative, logged) · `BLOCKED`

## Backlog (prioritized)

### A. Pipelining / inter-dispatch overhead (the ~20–25% gap)
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 1 | Fuse qwen q-norm+k-norm+rope-q+rope-k → 1 T1 kernel (precedent: gemma4_qkv_norm_rope_partial_t1) | HIGH | TODO | 4 tiny T1 dispatches/layer → 1 |
| 2 | Inter-dispatch fence audit (qwen decode) — find more redundant barriers | HIGH | TODO | +90% win was this class |
| 3 | ICB / multi-layer command-buffer batching — measure the "expected but unmeasured" ICB win | HIGH | TODO | inference/silicon/icb_recorder |
| 4 | Megakernel: fuse whole decode layer into one persistent kernel | MED | TODO | ambitious |
| 5 | Fuse split_packed (2 dispatches) into QKV matvec epilogue | MED | TODO | |
| 6 | Fuse kv_cache_write into rope-k kernel | MED | TODO | |
| 7 | Fold final RMSNorm into last layer encoder | LOW | TODO | |

### B. Batched GEMM / seq>1 (the big architectural lever)
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 8 | simdgroup_matrix (MMA 8×8) batched GEMM for seq>1 | HIGHEST | TODO | unblocks prefill+spec+DeepSeek-T>1 |
| 9 | Prefill attention path T>1 (qwen + MLA) | HIGH | TODO | DeepSeek T>1 hangs today |
| 10 | Spec-decode e2e (draft+target) after #8 | HIGH | TODO | 1.6–1.9× accept ceiling |
| 11 | Chunked prefill using fixed mha_causal (PR #54) | MED | TODO | |

### C. DeepSeek V2-Lite (stated focus)
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 12 | Fix kv_b de-interleave numerics (vs HF ref) | HIGH | TODO | dev-deepseek-e2e incoherent |
| 13 | Fix RoPE interleave pairing vs HF | HIGH | TODO | |
| 14 | Localize GPU OOB (Metal validation layer) → fix 30–40s/tok | HIGH | TODO | |
| 15 | Promote Q4_K MoE in-tree (~4× proven) | MED | TODO | |
| 16 | DeepSeek prefill path | MED | TODO | depends #9 |

### D. KV-cache
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 17 | KV-cache quantization (Q8_0 K/V) | HIGH | TODO | memory + long-ctx BW; 14B cache_max pinned 4096 |
| 18 | KV-cache Q4 + per-channel scales | MED | TODO | |
| 19 | Paged/ring KV cache for long context | MED | TODO | |

### E. Quant kernels / fit
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 20 | Consolidate 3 q3k_matvec ports → 1 canonical (fit enabler) | MED | TODO | 4B/8B/14B agents each wrote one |
| 21 | Canonicalize q5k_matvec (fit enabler) | MED | TODO | |
| 22 | Q3_K SoA-repack (4-aligned, vectorized) — parity experiment | LOW | TODO | best-case ~parity |
| 23 | IQ2_XXS matvec for extreme fit | LOW | TODO | |

### F. SSM
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 24 | mamba2_ssd.metal numeric fix (softplus/dt·B·x/D·x/n_groups) | MED | TODO | KNOWN BROKEN |
| 25 | mamba3 seqlen>1024 hang fix | LOW | TODO | |

### G. Structure / design / organizing
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 26 | Unify per-family launcher pattern (shared encode_* lib) | MED | TODO | qwen/gemma/deepseek |
| 27 | Centralize encode_quant_gemm PSO-dispatch across families | MED | TODO | |
| 28 | docs/kernels.md architecture index refresh | LOW | TODO | |
| 29 | best.md scoreboard refresh | LOW | TODO | |
| 30 | registry.py dtype-aware variant auto-derivation | LOW | TODO | |
| 31 | Unify mixed-quant detection across families (loader) | MED | TODO | |
| 32 | Generalize ICB recorder across families | MED | TODO | depends #3 |

### H. Sampling / head / norm / misc
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 33 | Fused argmax+sampling on-device (extend ICB) | LOW | TODO | |
| 34 | LM-head split-K for huge vocab | MED | TODO | |
| 35 | rmsnorm_t1 fusion variants for more dtypes | LOW | TODO | |
| 36 | RoPE table precompute/caching audit | LOW | TODO | |
| 37 | memory_aware_cache_max per-model headroom refinement | LOW | TODO | |
| 38 | fp16-fallback elimination audit (anywhere host-dequant streams 16bpw) | MED | TODO | the trap class |
| 39 | split-K decode gating tuning (extend the landed win) | LOW | TODO | |
| 40 | GQA amortization audit (mha_causal) — guard against regressions | LOW | TODO | |

## Log (newest first)
- **[wave-1 launch]** 5 worktree-isolated agents in flight (push own branch to remote; PR-only to main; bench derek/amelia, lexie down):
  - #1 Fused qkv-norm-rope T1 kernel — `ae697bf8e4d955340` — branch `dev-sk-qkv-rope-fusion`
  - #8 Batched GEMM (simdgroup_matrix MMA, seq>1; prefill/spec/DeepSeek-T>1 unblock) — `a0494e7e14793d0b5` — `dev-sk-batched-gemm`
  - #12–14 DeepSeek-V2-Lite numerics→coherence + GPU-OOB (base dev-deepseek-e2e) — `a6583284df8beb960` — `dev-sk-deepseek-fix`
  - #17 KV-cache quantization Q8_0 (short+long-ctx bench) — `a8c90d39fb2f2408a` — `dev-sk-kv-quant`
  - #3 ICB multi-layer batching (measure the unmeasured ICB win) — `a629b5aecd7b79de6` — `dev-sk-icb-batch`
- **[init]** Campaign branch `dev-sk-redesign` created off main 49f9174. PROGRESS.md seeded with operating model + 40-item backlog.
