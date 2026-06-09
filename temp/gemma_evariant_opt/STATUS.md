# gemma4 E-variant decode-opt hunt — STATUS

Branch: `dev-sk-gemma-evariant-opt-v2` (off origin/main @ fd6a95c). Bench host: amelia (M4 base, 16GB, colima VM present → timing has some contention).

## Headline finding: the DeepSeek anti-pattern does NOT exist in gemma E-variants — already fixed on main (#78).
The full Q8_0 body path (`SK_GEMMA4_BODY_Q8`, default ON) routes QKV/o_proj/gate/up/down through `q8_0_matvec_bf16` for E2B/E4B. No projection takes an fp16-gemm-M=1 or host-dequant route. The "bf16 body" my first analysis saw was the stale `dev-sk-redesign` worktree, not main.

## Measured (amelia, E2B, greedy, 63-tok decode, min-of-reps)
| build | decode tok/s | coherence |
|---|---|---|
| Q8 body + Q8 head (main default) | **20.88** | OK (Rayleigh answer, 60 tok) |
| bf16 body (SK_GEMMA4_BODY_Q8=0) | 13.40 | OK |
| bf16 LM head (SK_GEMMA4_LMHEAD_Q8=0) | 17.69 | OK |
| Q8 body + **fused q8_0_geglu (my kernel)** | **20.91** | OK (identical text) |

- Body Q8 already buys **1.56x** (13.40→20.88). Landed on main, not my work.
- LM head Q8 vs bf16 = **+3.2 tok/s** (large-N matvec IS bandwidth-bound, scales with bytes).
- My fused `q8_0_geglu_bf16_m1` (merge gate+up+gelu into 1 dispatch, −2 dispatch/layer, x read once): numerically correct (identical output) but **+0.03 tok/s = null**.

## Why fusion is null → the real bottleneck
- E2B Q8: 1.88 GB weight bytes/token, 47.8 ms/token → **39 GB/s effective = 33% of the ~120 GB/s M4 ceiling.**
- Removing ~70 dispatches/token moved nothing ⇒ NOT dispatch-count-bound.
- Same weight bytes after fusion ⇒ NOT raw-bandwidth-bound at the body matvecs.
- Conclusion: the small-N body matvecs (gate/up N=6144, QKV N≤2048, PLE N=256) are **occupancy/latency-bound** — they don't saturate the GPU at M=1 regardless of dispatch merging. The large-N LM head (N=262144) IS bandwidth-bound (hence the Q8-head win).
- Weight-quant lever is therefore **exhausted at Q8** for the E2B body: Q4_K body would cut bytes but adds heavier dequant, and we're already under the q3k matvec peak (67 GB/s) effective rate → no bandwidth headroom to recover. Matches the CLAUDE.md "Q4_K is the sweet spot, sub-Q4 is compute-bound net loss" and the M4-base "decode is GPU-wait/occupancy-bound" findings.

## The fused kernel (`q8_0_geglu_bf16_m1`)
Kept as a correct, slightly-leaner option (fewer dispatches, no m_up_scratch round-trip, env `SK_GEMMA4_DISABLE_Q8_GEGLU` to A/B). NOT a measurable win on this M4 → NOT landing it as a perf PR. Documented as a tried-and-null lever.

## Remaining candidate levers (not yet landed)
1. **Q4_K LM head** — the single bandwidth-bound matvec (0.43GB Q8 → 0.23GB Q4_K). ~0.2GB/token. Risk: logit accuracy on the tied embed/head. Worth a coherence-gated try.
2. **E4B paging check** — E4B (9.86) may be swap/wired-bound on 16GB; a fit lever (further quantize a resident-heavy tensor) could help there even though E2B is at ceiling. Needs E4B weights (~15G; amelia at 12G free — download blocked until disk freed).

## Discipline notes
- Worktree got flipped to `dev-sk-redesign` mid-session; my WIP was auto-preserved on my branch as commit `442a574`. Switched back. Did NOT touch dev-sk-redesign.
- Build: clang++ dylib on amelia (CLT-only, no xcrun metal) + SK_METAL_SRC_FALLBACK 23-source colon-list (runtime metal-compile). Verified PSO resolution (no UNRESOLVED/NaN).
