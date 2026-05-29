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

## Step 2 — spec loop: (status filled in as implemented)
## Step 3/4 — bench on derek (4B-Q4_K_M target + 0.6B-Q8 draft): (numbers below)

## Bench host
derek (M4 base, ~120 GB/s, load ~1.8, CommandLineTools-only — reuse prebuilt
libsk.metallib, relink C++ dylib only). GGUFs in ~/qwen-gguf/:
Qwen3-0.6B-Q8_0.gguf, Qwen3-4B-Q4_K_M.gguf.
