# Spec-decode on the batched-GEMM branch — STATUS

Branch `dev-sk-spec-decode` (base: gemm_mma seq>1 from dev-sk-batched-gemm / PR #55,
mha_causal chunk-fix #54). Goal: prove speculative decoding is now a WIN (was
0.54x in the May spike, before batched GEMM existed).

## What changed since the 0.54x spike (both gates the spike flagged are now lifted)
- **Gate 1 (correctness): mha_causal chunked-forward interior logits** — FIXED by
  PR #54 (49f9174). The spike saw gross wrong rows in a Br=2/parity pattern.
- **Gate 2 (no amortization): seq>1 forward** now routes QKV/V/O/gate/up/down
  through `gemm_mma` (commit 53da587): one weight read amortizes across M rows,
  so a seq=K verify costs << K x a seq=1 decode. The spike's blocker.

## Plumbing (already on the branch, from WIP bff4270)
- `sk_qwen_get_logits_rows(h, out_fp16, n_rows)` — all-position logits from the
  last forward (verify reads every K+1 row). launcher.c++ / launcher.h / qwen.py.
- `sk_qwen_get_pos` / `sk_qwen_set_pos` — KV cursor get/set to rewind rejected
  verify tokens.

## Orchestration (temp/spec_decode/)
- `spec_decode.py` / `specbench.py` — draft proposes K greedily; target verifies
  [cur,p0..p_{K-1}] in ONE seq=(K+1) forward (through gemm_mma); accept longest
  matching greedy prefix; first mismatch takes target's own token; rewind both
  KV cursors. specbench.py is the nohup-bench entrypoint (flushed per-line).
- `verify_logits_check.py` — Gate-1 regression: verify-mode per-row argmax vs
  autoregressive ground truth across prompt_len {1,7,16,17,31,33} x K {2,4,6}.
- `time_forward.py` — per-seq forward timing (gemm_mma on/off A/B).
- `_env.py` — local SK_DYLIB/SK_METALLIB shim (worktree build/).

## LOCAL validation (M4, qwen3-0.6B-Q8_0 — the only GGUF on this Mac)
1. **ABI works**: get_logits_rows returns (n_rows, vocab) fp16; get/set_pos OK.
2. **Gate 1 PASS (to fp tolerance)**: 70/72 verify rows match autoregressive
   exactly; the 2 "mismatches" are near-TIES (top-2 logits within 0.06; max
   |verify-auto| logit diff 0.22 — gemm_mma vs matvec accumulation order, the
   commit's documented <=1e-3 rel). NOT the spike's structural bug. The mha_causal
   interior-logits fix is confirmed working.
3. **LOSSLESS PASS**: spec-decode (draft==target==0.6B, K=4, n=32) produced ids
   IDENTICAL to plain greedy decode. accept/4=1.91, tok/fwd=2.91.
4. **gemm_mma amortization confirmed structurally** (per-seq forward, 0.6B):
       seq   gemm_mma ON   gemm_mma OFF (matvec loop)
       1       11.95 ms        14.28 ms
       2      112.21          43.53
       4      125.82          84.39
       6      130.66         129.69
       8      146.73         172.39
   OFF is ~linear (~21 ms/tok, the spike's no-amortization). ON is flat from
   seq2->8 (112->147) — amortized — BUT has a big fixed-cost floor that loses
   below seq~7 on 0.6B because its matrices (d_model=1024,n_int=3072) are too
   small to cover the BM=8 tile + threadgroup-staging overhead.

## WHY local can't show the perf win (and the bench MUST run on a host)
0.6B is the wrong scale: draft cost ~= target cost (spec-decode needs target>>draft),
and 0.6B matrices are too small for gemm_mma to beat the matvec below seq~7. The
real claim — 4B-Q4_K_M (d_model=2560,n_int=9728, ~4x dims) verify approaching
seq=1 cost — needs the 4B target, which is NOT on this Mac. Commit 53da587
measured 4B-class Q8_0 prefill 2.3-2.6x and seq=128 only 8-13x a seq=1 (sub-linear).

## BENCH DONE — lexie M4, qwen3-4b-q4km target + qwen3-0.6b draft (2026-06-08)
Build: locally-built dylib+metallib from a21fd25, rsync'd to lexie:~/sk_specdecode/
(self-contained, SK_DYLIB/SK_METALLIB pinned; did NOT touch lexie's own checkout).
GGUFs symlinked into the synced model_weights. n=64, reps=5 median, cache_max=1024.
macOS has no flock/setsid: used an mkdir-mutex /tmp/sk_bench.lock + an orphaning
subshell `( cmd >log 2>&1 </dev/null & )` (nohup fails "can't detach from console"
under non-tty ssh). Poll loop bounded <300s/ssh.

    prompt       K  baseline  spec    ratio   accept/K  tok/fwd  lossless
    creative     2  39.31     14.29   0.364   0.88      1.88     True
    creative     4  39.46     18.68   0.473   2.05      3.05     True
    creative     6  39.47     19.39   0.491   2.76      3.76     True
    low-entropy  2  39.40     15.20   0.386   0.97      2.00     True
    low-entropy  4  39.34     24.49   0.623   2.94      4.00     True
    low-entropy  6  39.40     25.36   0.644   3.85      4.92     True

LOSSLESS = True on all 6 (spec greedy ids == target greedy ids exactly). Accept
rates are HEALTHY (up to 4.92 tok/fwd at K=6). Best ratio 0.644x — better than the
0.54x May spike but STILL < 1.0. NOT A WIN.

## WHY it loses — the gemm_mma seq>1 verify does NOT amortize on M4
time_target.py (per-seq forward timing, 4b-q4km on lexie):
    target seq=1 decode : 25.44 ms  (= 39.3 t/s, == baseline)
    draft  seq=1        :  6.96 ms
    K=2 verify seq=3 : 116.34 ms  (4.6x a seq=1, vs ideal ~=1x)  break-even accept 4.12
    K=4 verify seq=5 : 132.65 ms  (5.2x)                          break-even accept 5.31
    K=6 verify seq=7 : 147.86 ms  (5.8x)                          break-even accept 6.45
Spec step cost = K*draft_seq1 + target_verify. To beat baseline you need
mean-accept >= break-even, but the break-even accept EXCEEDS K at every K
(4.12>2, 5.31>4, 6.45>6) — so a win is mathematically UNREACHABLE here regardless
of acceptance. Root cause: the gemm_mma seq>1 path has a large fixed-cost floor on
M4 — a seq=3 forward costs ~4.6x a seq=1 decode instead of ~1x. The #55 prefill
win (2.3-2.6x) was at seq=128 and on Q8_0; at the tiny seq=K+1 verify lengths
spec-decode uses, the matvec decode path is far cheaper per token. Same floor seen
on 0.6B locally (seq2 = 112 ms vs seq1 = 12 ms).

## 8B follow-up — NOT RUN (no host slot)
amelia has 8B-Q4_K_M but NO 0.6B draft GGUF and ~58 MB free (load 2.1, rising).
derek has the 0.6B but only 8B-Q8 (8.7G, won't fit w/ draft in 16G), load 8.18,
~69 MB free. Neither can host 8B+draft now. Prediction: an 8B target raises seq=1
cost (~2x) which shrinks the floor's relative weight, but the verify forward
inherits the SAME non-amortizing gemm_mma floor, so a win is unlikely to flip
without fixing that kernel's small-seq cost.

## VERDICT — LOSSLESS GREEN, PERF NOT A WIN (0.36-0.64x)
Speculative decoding is now provably LOSSLESS and the orchestration/ABI are
correct end-to-end on a real 4B target. But it does NOT beat baseline decode on
M4: 0.644x best, and break-even accept > K means no K tuning recovers it. The
blocker is no longer "no batched GEMM" (that exists) — it is that the gemm_mma
seq>1 verify forward does not amortize at the small seq lengths (K+1 <= 7) that
spec-decode generates on M4; it costs ~5x a seq=1 decode. NO PR for a perf win.
The real unblocking lever is a verify-forward kernel whose seq=2..8 cost
approaches seq=1 (a low-fixed-cost batched matvec / a wider-decode kernel), not
more spec-decode tuning. Until then spec-decode is a regression on M4.
