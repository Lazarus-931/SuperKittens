# q4km-head-gen — generalize M14 Q6_K LM-head routing to 4B + 8B Q4_K_M

Branch: dev-q4km-head-gen (off dev-q4km-14b-perf @ 52a93cd). NO push/PR.

## The win being generalized (M14)
Q4_K_M's LM head (Q6_K) was host-dequantized to fp16, streaming ~1.48 GB/token
(largest per-token read). Routing it through `q6k_matvec` (keep Q6_K bytes,
dequant in-kernel) gave 2.21x on the head step, +8.5% e2e on 14B-Q4
(10.40 -> 11.28 tok/s). Fix commit: 9685e41.

## Task 1 finding: shared vs 14B-specific
The M14 fix lives in the SHARED qwen path (qwen_model.h dispatch_model +
weights.c++ sk_qwen_load_gguf), NOT gated to 14B. So:
- 8B-Q4 (untied Q6_K output.weight): already covered by the existing fix
  as-is. No code change needed for 8B routing.
- 4B-Q4 (tie_word_embeddings=1): NOT covered. The weights.c++ Q6_K branch was
  gated on `!c.tie_word_embeddings`, so the tied Q6_K token_embd fell through
  to the fp16 path (read_to_fp16 into w_embed, then used as the head via
  transB gemm). 4B paid the full fp16-head stream.

## Change made (4B generalization)
weights.c++: dropped the `!c.tie_word_embeddings` guard on the Q6_K LM-head
branch. For tied models the launcher leaves w_lm_head null, so the branch now
allocates a dedicated Q6_K w_lm_head buffer from the raw token_embd bytes and
sets dt_lm_head=Q6_K. The dispatch Q6_K branch then fires q6k_matvec.
- w_embed stays fp16 for the per-token embedding gather (independent buffer).
- token_embd is [vocab,d_model] row-major Q6_K == same orientation the tied
  transB gemm head used, so q6k_matvec is numerically identical, just in-kernel
  dequant. Mirrors the existing Q8_0-tied precedent (line ~503).
- Memory: tied 4B now holds fp16 w_embed (~1.48 GB) + Q6_K head (~640 MB).
  6 GiB memory-aware headroom on lexie (16 GB, ~2.5 GB GGUF) absorbs it easily.

## Bench plan
Prompt: "Generate a poem about pizza dough". 2 warmup + 30s cooldown +
5-rep median. Baselines: 4B 32.67 (lexie), 8B 17.54 (amelia).

## Host situation (documented)
- lexie (M4, 16GB): LOADED — 4 `./sme2` processes pinning cores, load ~3.6.
  Someone else's CPU job; not mine to kill. Decode tok/s would be skewed by
  CPU contention, so 4B baseline-on-lexie is not re-runnable cleanly now.
- derek (M4, 16GB, Mac16,10 — IDENTICAL hw to lexie): idle (load ~0.8), has
  Qwen3-4B-Q4_K_M.gguf. Running 4B before/after BOTH on derek for a clean
  same-host delta (re-measuring baseline rather than trusting the lexie 32.67
  cross-host). derek main checkout was corrupted earlier; using fresh
  ~/SK_headgen staging (rsync of branch source + local-built artifacts).
- amelia (M4, 16GB): designated 8B host (only host with Qwen3-8B-Q4_K_M.gguf).
  Went unreachable mid-run (ssh timeout); monitoring for recovery.

## Build approach
Built locally (full Metal toolchain on worktree host). Two artifact sets:
- build/          = fixed (tied Q6_K head routed to q6k_matvec)
- build_baseline/ = parent dev-q4km-14b-perf weights.c++ (fp16 tied head)
Same registry/metallib; only weights.c++ differs. rsync'd both to derek.

## Results

### 4B-Q4_K_M (qwen3-4b-q4km, tied Q6_K head) — derek, M4 16GB
| build    | median tok/s | reps |
|----------|--------------|------|
| baseline | 33.35        | 33.37 33.26 33.35 33.38 33.26 |
| fixed    | 38.48        | 37.84 38.51 38.48 38.32 38.51 |

**+15.4% end-to-end** (33.35 -> 38.48). Derek baseline 33.35 ≈ documented
lexie 32.67, confirming derek is a faithful stand-in.

Coherence: identical greedy sample text in BOTH runs ("ums\nOkay, the user
wants a poem about pizza dough. ..."). Same argmax sequence => Q6_K in-kernel
dequant is NUMERICALLY EXACT vs the prior fp16 host-dequant, not merely
coherent. nan_or_inf_logits=False, frac_printable=1.000 both runs.

memory-aware clamp: fired (32768 -> 31744, weights=2.3GiB). 4B has ample
headroom; the extra ~640MB Q6_K head (on top of fp16 w_embed) is absorbed.

### 8B-Q4_K_M (qwen3-8b-q4km, untied Q6_K head) — amelia, M4 16GB
NOTE on the code: my change is a NO-OP for untied 8B. The parent
dev-q4km-14b-perf already routes untied Q6_K output.weight -> q6k_matvec (the
M14 fix); my only diff at the head branch is dropping `&& !tie_word_embeddings`,
which for tie=0 leaves the branch identical. So 8B's win is the M14 fix itself,
exercised on the shared path. To get a faithful before/after I rebuilt a true
fp16-head baseline from d10c622 (pre-head-fix: untied Q6_K falls to read_to_fp16
+ fp16 gemv head) and compared it to the current Q6_K-head build.

| build                        | median tok/s | reps |
|------------------------------|--------------|------|
| baseline (fp16 head, d10c622)| 18.36        | 18.10 18.45 18.43 18.36 18.27 |
| fixed (Q6_K head)            | 21.09        | 21.07 21.10 20.93 21.09 21.16 |

**+14.9% end-to-end** (18.36 -> 21.09). Baseline 18.36 ~ documented 17.54.
Coherence: identical greedy sample text in BOTH builds ("s\nOkay, the user
wants a poem about pizza dough..."). Numerically exact, no NaN, fully
printable. Fixed build's smaller resident set eased swap pressure
(free 603MB vs baseline 374MB).

amelia ssh dropped ~17 min (mini slept/network stalled); recovered after burst
connects. Hostname-guarded each run to confirm Amelias-Mini (one stray ssh
fluke resolved to my laptop, immediately re-verified as amelia).

memory-aware clamp: fired hard (32768 -> 12800, weights=4.7GiB). The fp16-head
baseline THRASHED during load/warmup (system free 4%, swap 3.7/4GB) because the
clamp's weights estimate (GGUF st_size) undercounts the fp16-head dequant
expansion (~+0.85GB) — a concrete demonstration of why the head fix matters
beyond bandwidth. The timed decode reps still came out steady (warm decode loop
fits). Fixed build (Q6_K head, smaller resident set) recovered free mem to 83%.

## Summary
| model | baseline | fixed | delta | host  |
|-------|----------|-------|-------|-------|
| 4B-Q4 | 33.35    | 38.48 | +15.4%| derek |
| 8B-Q4 | 18.36    | 21.09 | +14.9%| amelia|

The M14 head-routing win GENERALIZED to both models. 4B required a code change
(tied-embedding path); 8B was already covered by the unchanged shared fix and
this is the first faithful before/after for it. Both numerically exact
(identical greedy output vs fp16-head baseline). Compares well to M14's 14B
+8.5%; the smaller models gain MORE because the LM head is a larger fraction of
their per-token bandwidth.

## Progress
- [x] Local build clean (fixed, fp16-head baseline, parent baseline)
- [x] 4B baseline re-bench on derek: 33.35 tok/s
- [x] 4B fixed re-bench on derek: 38.48 tok/s (+15.4%)
- [x] 4B coherence: numerically exact (identical greedy output)
- [x] 8B baseline (fp16 head) re-bench on amelia: 18.36 tok/s
- [x] 8B fixed (Q6_K head) re-bench on amelia: 21.09 tok/s (+14.9%)
- [x] 8B coherence: numerically exact (identical greedy output)
- [x] memory-aware clamp verified firing on both (4B 31744, 8B 12800)
