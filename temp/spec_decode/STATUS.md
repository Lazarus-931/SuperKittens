# Spec-decode spike — STATUS

Branch `dev-spec-decode` off `origin/dev-q4km-e2e` (d10c622). Worktree, no push/PR.

Goal: draft (qwen3-0.6B-Q8_0) proposes K tokens greedily; target (qwen3-4B-Q4_K_M)
verifies all K in one forward pass; accept longest matching greedy prefix.
Amortize one target weight-read over up to K accepted tokens.

## Step 1 — can the target return ALL-position logits? (the gating question)

Findings from reading the qwen launcher + dispatch_model (qwen_model.h):

- **GPU already computes all-position logits.** `B.logits` is allocated
  `T_max * vocab_size * fp16` (launcher.c++ ~L180). The LM-head GEMM (qwen_model.h
  ~L989-1052) writes one full V-row per input position, at RELATIVE rows `0..T-1`
  (offset `m * N_v`), for the F16 GEMM, the M==1 gemv, and the Q8_0 per-row loop.
  So a K-token forward produces K full logit rows in `B.logits`.
- **SK did NOT expose them.** Two blockers:
  1. `sk_qwen_get_last_logits` reads absolute cache row `current_pos - 1`
     (launcher.c++ ~L437-444). For a multi-token prefill it returns the wrong row
     unless current_pos happens to equal K, and only ever returns ONE row.
  2. `output_id` buffer is `batch * int32` only (launcher.c++ L177). The `argmax`
     kernel dispatches T threadgroups writing `out[row]` for row 0..T-1
     (sample.metal L75), so for K>1 it writes past the buffer — i.e. the C ABI
     never surfaced per-position argmaxes either.

**Verdict on step 1: the all-position logits exist on-device; the only gap is a
read-back ABI. Small fix, NO kernel change, NO metallib rebuild.**

### The SK change (this spike)
- C ABI `sk_qwen_get_logits_rows(h, out_fp16, n_rows)`: memcpy the first `n_rows`
  RELATIVE rows of `B.logits` (each `vocab_size` fp16). Argmax done in numpy for
  the spike (correctness over speed; the per-row argmax kernel can be wired later).
- Python `Qwen.get_logits_rows(n_rows) -> (n_rows, vocab) fp16`.

## Step 2 — spec loop: IMPLEMENTED (temp/spec_decode/spec_decode.py)
Draft proposes K greedily; target verifies [cur, p0..p_{K-1}] (K+1 rows) in one
forward; accept-longest-greedy-prefix; first mismatch takes target's argmax;
KV cursor rewound (sk_qwen_set_pos) to drop rejected tokens. Both handles share
the qwen3 tokenizer (same 151936 vocab → ids interchangeable).

## Step 1b — SECOND blocker found during validation (verify logits wrong)
get_logits_rows works, BUT the underlying multi-token forward at current_pos>0
(the verify scenario) returns WRONG logits at some interior query rows.
- A single forward over the WHOLE seq at current_pos=0 (pure prefill): ALL rows
  correct (verified vs autoregressive T=1 ground truth).
- A multi-token forward AFTER the KV is non-empty (verify): interior rows wrong
  in a Br=2 / current_pos-parity pattern (e.g. prompt_len=17 → row 3 wrong at
  K=4; at offset 15 rows 1 and 3 wrong). Data-dependent (argmax sometimes
  survives the bad attention, so it's intermittent).
- Root cause is in `mha_causal` (kernels/attn/attn.metal, fa_d128 template):
  the Br=2 TG-staged kernel mishandles the causal window for some query rows
  when kv_len > seq (chunked decode with prior KV). CLAUDE.md flags mha_causal
  as production-correct and warns against edits — NOT touched in this spike.
  EXACT FIX NEEDED: correct the per-q_row causal limit / KV stride in fa_d128
  for the kv_len>seq case, then re-verify multi-row argmax == autoregressive.

## Step 3/4 — BENCH (derek, M4 base, cache_max=1024, 5-rep median, n=64, K=4)
                         baseline   spec-decode   mean accept/4   tok/target-fwd
  creative   (poem)      34.37 t/s   18.45 (0.54x)    1.62            2.67
  low-entropy(primes)    34.30 t/s   19.13 (0.56x)    1.91            2.78

Spec-decode is SLOWER (0.54-0.56x), and outputs diverge (verify-kernel bug:
"Let me me think think"). Acceptance is moderate (1.6-1.9 of 4) even on the
low-entropy prompt, but that's not why it's slow.

## THE decisive finding — per-op timing (derek)
  target fwd seq=1:           29.3 ms   (= ~34 tok/s baseline)
  target fwd seq=5 (verify):  114.6 ms  (3.9x slower than seq=1 — NOT amortized!)
  draft  fwd seq=1:            7.0 ms
  get_logits_rows(1)+argmax:   0.20 ms  (host copy negligible)

The spec-decode premise ("1 target weight-read amortizes K tokens") DOES NOT
hold in SK's qwen forward: a seq=5 forward costs ~3.9x a seq=1 forward, so the
K verify positions re-pay per-token cost instead of riding one weight read.
Per K=4 round: 4 draft (28ms) + 1 verify seq=5 (114ms) = 142ms / ~2.7 accepted
= 52 ms/token vs 29 ms/token baseline → 0.54x. Fully explains the slowdown.

WHY no amortization: at seq=1 decode is bandwidth-bound (weights dominate, one
read). At seq>1 the per-position compute scales with seq — the LM-head GEMM is
M_v=T rows over 151936 vocab, attention is ~seq*kv, and the Q4_K/Q8 matvec
kernels are matvec (M=1) loops dispatched per-row. SK has no seq-batched GEMM
fast path for the LM head or the quant matmuls, so a 5-token forward ≈ 5 stacked
1-token forwards in compute, defeating amortization on the bandwidth ceiling.

Forward cost vs seq (derek, 4B-Q4_K_M):
  seq=1  29.7 ms  (29.7 ms/pos)
  seq=2  55.3 ms  (27.6)
  seq=3  74.7 ms  (24.9)
  seq=4  94.4 ms  (23.6)
  seq=5 114.1 ms  (22.8)
  seq=8 174.0 ms  (21.8)
Per-position cost falls only 29.7 -> 21.8 ms (1.36x) from seq 1 -> 8. The
amortizable weight read is <30% of per-token cost; the rest is per-position
compute that scales with seq. For spec-decode to win, a seq=K forward must cost
~= a seq=1 forward; here it costs ~K * 0.75 * (seq=1). No win available.

## VERDICT
Speculative decoding is NOT worth building into SK as-is. Two hard gates:
 1. Multi-token forward at current_pos>0 returns wrong interior logits
    (mha_causal Br=2 / kv_len>seq bug) — correctness blocker.
 2. Even fixed, the seq>1 forward is not weight-read-amortized (3.9x at seq=5),
    so there is NO speedup to capture until SK gets seq-batched GEMM fast paths
    for the LM head + quant matvecs (turning the verify into ~1 weight read).
Realistic speedup TODAY: <1x (regression). Potential AFTER (1)+(2): bounded by
mean-accept (1.6-1.9) → ~1.6-1.9x ceiling, and only if a fixed seq=K+1 forward
approaches the cost of a seq=1 forward. That GEMM work is the real prerequisite;
spec-decode is a thin orchestration layer on top of it.
