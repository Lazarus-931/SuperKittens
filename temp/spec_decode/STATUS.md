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

## BENCH RESULT (lexie, M4 16GB, qwen3-4b-q4km target + qwen3-0.6b draft, N=64, reps=5)
Baseline plain 4b-q4km decode = 39.4 tok/s (25.3 ms/token). LOSSLESS=True everywhere.
| prompt      | K | spec t/s | ratio | accept/K | tok/fwd |
|-------------|---|----------|-------|----------|---------|
| creative    | 2 | 21.98    | 0.557 | 0.88     | 1.88    |
| creative    | 4 | 25.45    | 0.645 | 2.05     | 3.05    |
| creative    | 6 | 26.50    | 0.672 | 2.76     | 3.76    |
| low-entropy | 2 | 23.33    | 0.592 | 0.97     | 2.00    |
| low-entropy | 4 | 33.38    | 0.849 | 2.94     | 4.00    |
| low-entropy | 6 | 34.61    | 0.882 | 3.85     | 4.92    |

STILL A LOSS (0.56-0.88x), though improved from the old 0.36-0.64x. Lossless holds.

## WHY it still loses (time_target.py + time_dbg.py diagnostics)
- target seq=1 decode = 25.3 ms; draft 0.6B seq=1 = 6.92 ms (27% of a target decode).
- verify forward floor is too high: seq=3 = 63.7 ms (2.53x a decode), seq=5/7 = 76 ms.
  The small-M kernel helps ONLY at seq=3 (63.7 vs MMA 74.9 ms); at seq>=5 Q4_K routes
  to MMA (gemm_sm_wins Q4_K only M<=4) and both plateau at ~76 ms. A 4B-Q4KM decode is
  bandwidth-bound (~2.5GB weights/token); the multi-row forward is dequant/compute-bound
  so it does NOT collapse to ~1x a decode the way the isolated GEMM microbench implied.
- all-rows LM head adds 6.6/13.0/19.5 ms at seq 3/5/7 (K+1 Q6_K full-vocab projections).
- break-even mean-accept: K=2 -> 2.32 (impossible), K=4 -> 3.61, K=6 -> 4.42. Actual
  accept tops out at 3.85 (low-entropy K=6) < break-even. Draft tax (K*6.92 ms) dominates.

## K sweep (lexie, same build/host) — peak is still < 1.0x
| prompt      | K | ratio | accept/K |
|-------------|---|-------|----------|
| creative    | 3 | 0.507 | 1.13     |
| creative    | 5 | 0.688 | 2.50     |
| creative    | 8 | 0.423 | 2.56     |
| low-entropy | 5 | 0.784 | 2.94     |
| low-entropy | 8 | 0.640 | 4.25     |
Best across all K = 0.88x (low-entropy K=6). K=8 regresses (draft tax + verify floor
grow faster than accept; Q4_K verify M>4 routes to MMA, no small-M help).

## 8B-target projection (qwen3-8b-q4km + 0.6b) — likely the only path to >1.0x, BLOCKED
From measured 4B scaling: an 8B-Q4KM decode ~= 2x a 4B decode (~50 ms; ~2x weight
bytes, bandwidth-bound) while the 0.6B draft tax stays 7 ms/token, so the relative
draft tax HALVES. Verify(8B, seq7) ~= 2x verify(4B) ~= 150 ms. Break-even accept(K=6)
~= (6*7 + 150)/50 - 1 ~= 2.84, vs measured low-entropy accept 3.85 -> 8B should WIN
(~1.1-1.3x) at low-entropy. NOT confirmed: no 8B-Q4KM on lexie; amelia has it but is
memory-tight (0.4 GB free) + no 0.6B draft + no spec ABI/Xcode; inter-host 5GB copy
to lexie ran at ~40 MB/min (impractical). Infra-blocked, not code-blocked. To run:
put Qwen3-8B-Q4_K_M.gguf in a snapshot dir on an idle 16GB mini and
`specbench.py --target qwen3-8b-q4km --draft qwen3-0.6b --K 4 6 --cache-max 512`.

## CONCLUSION
Orchestration correct + LOSSLESS; small-M rebase improved spec-decode from 0.36-0.64x
to 0.56-0.88x, but it is STILL a net LOSS on M4 for the 4B target. No PR-for-win (it's
not a win). 8B target is the projected win but is infra-blocked on the current hosts.

## Local-only artifacts (NOT committed)
- `_snap_06b/` symlink snapshot, `local_lossless.py`, `repro.py`, `fit_check.py`,
  `time_dbg.py`, `draft_dbg.py` (kept, harmless local harnesses).
