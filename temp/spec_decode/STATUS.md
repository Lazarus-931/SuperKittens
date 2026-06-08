# Spec-decode on the small-M GEMM branch — STATUS

Branch `dev-sk-spec-decode-smallm` (base: `origin/dev-sk-batched-gemm` HEAD 2fca67f,
which has gemm_mma + gemm_mma_smallm (PR#59) + the M-threshold dispatch). Goal:
prove speculative decoding is now a WIN now that the verify-forward floor is fixed.

## Why this rebase exists
The OLD spec-decode bench (`dev-sk-spec-decode`, f722ab7) was LOSSLESS but 0.36-0.64x:
the gemm_mma seq<=7 verify forward cost 4.6-5.8x a decode (MMA BM=8 fixed floor).
PR#59 added `gemm_mma_q8_0_sm` / `gemm_mma_q4k_sm` (multi-RHS matvec, no MMA tile
floor) cutting the verify floor to seq2 1.0-1.1x, seq4 1.3-1.5x, seq8 2.3-2.8x.
Break-even mean-accept should now be below K -> spec-decode should WIN.

## What was ported on top of dev-sk-batched-gemm (additive only; kept its kernel/dispatch)
- ABI: `sk_qwen_get_logits_rows`, `sk_qwen_get_pos`, `sk_qwen_set_pos`,
  and NEW `sk_qwen_set_lm_head_all_rows` (launcher.c++/.h, qwen.py).
- **LM-head all-rows fix (load-bearing)**: PR#56 optimized prefill to project the
  LM head for row T-1 ONLY. Spec-decode verify needs per-position logits for all
  K+1 rows, so `ModelParams.lm_head_all_rows` (Handle field + setter) makes the
  head loop over rows 0..T-1. Default (0) keeps the PR#56 last-row-only fast path.
  Spec-decode harness calls `target.set_lm_head_all_rows(True)` after load.
- Verify forward (seq=K+1) auto-routes through gemm_mma_smallm: `gemm_sm_wins`
  (qwen_model.h) returns true for Q8_0 M<=8, Q4_K M<=4, so K=2,4,6 (seq 3,5,7) hit
  the small-M kernel for the per-layer GEMMs. Confirmed by build (PSOs present).

## LOCAL validation (M2 Air 8GB — 0.6B only; 4b target does NOT fit here)
- `smoke.py`: ABI present; all-rows verify rows bit-match per-position decode = OK.
- `local_lossless.py`: full loop, 0.6B as both target+draft, K=2,4,6 all
  `lossless=True`. Proves orchestration + get_pos/set_pos rewind + all-rows verify
  are end-to-end correct. (Run with SK_SNAP_06B pointing at a 0.6B Q8_0 gguf.)
- Build: `./build.sh` -> build/libsk.{dylib,metallib}. New ABI symbols exported,
  gemm_mma_q8_0_sm / gemm_mma_q4k_sm PSOs in the metallib.

## BENCH (on an idle 16GB mini — lexie/derek/amelia)
draft qwen3-0.6b + target qwen3-4b-q4km. specbench.py is the nohup entrypoint
(flush per line). Protocol: double-fork `( cmd </dev/null >~/spec_bench.log 2>&1 & )`,
poll w/ SHORT ssh every 60-90s, /tmp/sk_bench.lock mkdir-mutex. Clamp cache_max
(~1024) so draft 0.6B + target 4B-Q4KM both fit in 16GB.
Report: spec tok/s vs plain 4b-q4km decode (ratio per K), accept/step, lossless.
OLD = 0.36-0.64x; is it now >1.0x?

## Local-only artifacts (NOT committed)
- `_snap_06b/` symlink snapshot, `local_lossless.py` harness (kept, harmless).
