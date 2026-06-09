# DeepSeek-V2-Lite MLA T>1 prefill — "hang" is stale; REAL bug found+fixed: gemm_fp16 row-grid /64 vs BM=32 (dense-L0 MLP rows silently skipped at T>32)

Branch: dev-sk-ds-prefill-hang (base local main @71df9ea; deepseek tree == main pre-fix).
Host: amelia (M4, 16 GB, CLT-only; runtime metal-compile). Lab: ~/sk-ds-pfhang-r2.
Model: DeepSeek-V2-Lite Q4_K_M (~10 GB resident), seq_max=cache_max=512, ONE handle/process,
all runs detached + perl-alarm hard timeout, caffeinate -is.

## VERDICT (two-part)
1. **The historic "T>1 prefill hangs GPU" does NOT reproduce on main — stale note.**
   It was reported mid-PR #71 (a6e4972, 2026-06-08) and fixed inside that same PR
   (NULL c_kv/k_pe bindings → undefined GPU reads; head-major vs token-major Q
   strides in rope_interleave/mla_decode_v2). T=16/64/128 prefill all complete,
   logits finite, no command-buffer errors.
2. **But T>1 prefill on main was silently WRONG and NONDETERMINISTIC for T>32**
   (exactly the regime single-stream TTFT cares about). Root-caused + fixed
   (one-line dispatch-grid fix), now token-identical to tok-by-tok at every
   probed T, and the TTFT win is banked honestly on the FIXED build.

## The real bug (root cause, kernel/line/mechanism)
`kernels/gemm/fp16/gemm.metal` `gemm_fp16` tiles **BM=32 rows × BN=64 cols**
(retiled to 32×64×32 on 2026-05-03 @9c26277). DeepSeek's `encode_gemm`
(models/deepseek/deepseek_model.h:283, added 2026-05-11 @022fce3 — born buggy)
dispatched the ROW grid as `(M + 63) / 64`, i.e. one 32-row threadgroup per
**64** rows: at M=T>32, rows 32..63 of every 64-row block are NEVER COMPUTED.
On V2-Lite the only `gemm_fp16` user at T>1 is the **dense layer-0 MLP**
(gate/up/down, fp16, n_int=10944; everything else is quant→gemm_mma/matvec, and
M=1 decode uses gemv). Consequences, all reproduced and explained:
- **Skipped-row contents = stale scratch.** `mlp_gate/mlp_up/shared_out` are
  reused by every MoE layer's shared-expert MLP, so at layer 0 of call N the
  skipped rows hold call N-1's layer-26 shared-expert/residual values (|x| up
  to ~700) → `y_out = y_attn + stale` injects them into the residual stream.
- **Cross-call feedback loop**: each call's corruption changes the next call's
  stale rows → drift that AMPLIFIES per call. Measured per-call L0-output L2
  (SK_DS_DBGL, T=128): 54.8 → 2091 → 2932 → 3948 → 5029 → 6154 (runs 0-5);
  next-token sequence [4541, 4541, 31327, 31327, 46713, 46713, ...] —
  EXACTLY reproducible across processes (deterministic state evolution, not a
  timing race; persists under per-layer commit+wait).
- **First call after load is (accidentally) right**: buffers alloc_zero → the
  skipped rows read 0 → last rows just miss the L0-MLP contribution; argmax
  survived it at both T=64 (317 = " is", correct) and T=128 (4541) — the wrong
  tokens only appear once the skipped rows hold real stale data (call ≥2).
- **Why every earlier probe missed it**: T≤32 ⇒ (M+63)/64 == (M+31)/32 == 1 ⇒
  full coverage (short coherence probes T=6..11 were token-identical); M=1
  decode and N≤32 batched serving identical by the same arithmetic; tok-by-tok
  prefill never leaves M=1. The prior prefill campaign's "per-slot T>1 matvec
  has a latent bug / MMA is more correct (4/5 vs 1/5 probes)" observation was
  THIS bug — path-independent: the drift sequence is bit-identical under
  SK_DS_NO_MOE_MMA, SK_DS_NO_MMA_PROJ, both, and SK_DS_NO_LMHEAD_LAST.

## The fix (committed on this branch)
`models/deepseek/deepseek_model.h` — row grid `(M + 63) / 64` → `(M + 31) / 32`
in `encode_gemm` (live bug) and in the fp16 LM-head fallback (same latent
pattern; dead on V2-Lite whose head is Q6_K). Grids are IDENTICAL for M≤32 ⇒
decode (M=1) and batched serving (N=8 lanes ⇒ M=8) dispatch byte-identically;
only the broken T>32 prefill rows change (they go from "skipped" to "computed").
Cross-build decode-stream identity verified empirically (gate below).

Blast radius (flagged, NOT touched here): gemma4_model.h and qwen_model.h carry
`(M + 63) / 64` row grids on their fp16 GEMM call sites, and kernels/gemm/gemm.c++
:19 has the same arithmetic — gemma's body GEMM was retuned to BM=64 tiles so
/64 may be correct there; qwen prefill routes through gemm_mma. Each family
needs its own check against the kernel it actually binds. mamba2/3 already use
(M+31)/32.

## Gates (all PASS)
1. **Repro on main build**: NO hang at T=16/64/128 (detached + hard timeout,
   SK_DS_DEBUG finite logits). Hang = stale. Nondeterminism/wrong-rows = the
   real, newly diagnosed bug (12-run drift, see above).
2. **Fix correctness** (fixed build, one handle, fresh reset per side):
   - drift gate: 12 fresh seq T=128 forwards → 4541 × 12, == tok-by-tok truth.
   - next + 32-greedy continuation TOKEN-IDENTICAL (33/33) tok-by-tok vs T>1 at
     T=6, 11, 11 (real probes) AND T=64, 128, 256 (long synthetic). Continuations
     fluent, logits finite throughout.
3. **TTFT A/B** (fixed build, single stream, 2 warmup + 7 reps median):
   | T   | tok-by-tok | T>1 forward | speedup | faster | vs ≥8% bar |
   |-----|-----------|-------------|---------|--------|------------|
   | 128 | 2723.6 ms | 751.3 ms    | 3.63×   | 72.4%  | PASS (~9×) |
   | 256 | 5544.5 ms | 1230.6 ms   | 4.51×   | 77.8%  | PASS       |
   Rep spread <±0.3% on the seq side. Fixed-build T>1 cost == pre-fix broken
   build (751 vs 768 ms @T=128) — the missing row-tiles were skipped work the
   model NEEDED; computing them costs ~nothing (dense L0 is 1/27 layers).
4. **Decode byte-identity across builds**: 65-token greedy stream (tok-by-tok
   prefill, T=1-only dispatches) IDENTICAL base vs fixed, AND grid arithmetic
   identical for all M≤32 ⇒ single-stream decode + batched serving (N=8 ⇒ M=8)
   dispatch byte-identically by construction.

## Single-stream picture
`generate()`/`chat()` already prefill the whole prompt as ONE T>1 forward — on
main that path was silently corrupted for prompts >32 tokens (and got worse the
longer the process lived). The fix makes the fast path correct; the TTFT table
quantifies what T>1 buys vs the tok-by-tok fallback. The serving batched-lane
prefill port (generate_batched still feeds prompts at seq==1) remains the
follow-up item and now has a correct T>1 substrate to build on.

## Pre-fix A/B (main build, for the record; seq>1 side WRONG for T>32)
T=128: tok-by-tok 2724.0 ms → seq>1 768.4 ms = 3.55×; T=256: 5553.5 → 1224.8 ms
= 4.53×. Kept only as an upper-bound sanity reference; the honest numbers are
the fixed-build ones above.
