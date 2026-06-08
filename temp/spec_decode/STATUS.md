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

## BENCH STATE — READY-TO-BENCH (no host slot at attempt time)
- Required GGUFs present on **lexie**: 0.6B-Q8 (model_weights/Qwen3-0.6B/),
  4B-Q4_K_M (~/qwen-gguf/Qwen3-4B-Q4_K_M.gguf). amelia/derek lacked the 4B-Q4_K_M.
- All three minis 16GB and CONTENDED at attempt (load 1.2-1.8; free mem
  amelia 1.2G / derek 131M / lexie 1.35G). draft(0.6B ~0.6G)+target(4B-q4km ~2.5G)
  both resident needs headroom these didn't have. Prior identical agent was KILLED
  by the 600s watchdog on a synchronous contended-derek bench — did NOT repeat.
- lexie SK checkout is on dev-perf-iter w/ a stale (May 26) build.

### To run the bench (lexie, the only host with both GGUFs; least loaded):
1. Get this branch's build onto lexie. Either:
   (a) rsync the locally-built `build/libsk.dylib` + `build/libsk.metallib` to
       lexie (self-contained; avoids concurrent xcrun crashes), OR
   (b) on lexie: `git fetch && git checkout dev-sk-spec-decode && ./build.sh`.
2. Link 4B-Q4_K_M into model_weights so sk.load finds it:
   `mkdir -p ~/SuperKittens/SuperKittens/model_weights/Qwen3-4B-GGUF && \
    ln -sf ~/qwen-gguf/Qwen3-4B-Q4_K_M.gguf .../Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf`
   (tokenizer.json already in that dir for chat prompts).
3. Hold the lock around the bench only:
   `( flock -n 9 || exit 1; cd ~/SuperKittens; \
      SK_DYLIB=build/libsk.dylib SK_METALLIB=build/libsk.metallib \
      PYTHONPATH=. nohup python -u temp/spec_decode/specbench.py \
        --target qwen3-4b-q4km --draft qwen3-0.6b --n 64 --K 2 4 6 \
        --cache-max 1024 --reps 5 > ~/spec_bench.log 2>&1 & ) 9>/tmp/sk_bench.lock`
4. POLL every 60-90s with SHORT ssh: `ssh lexie 'tail -3 ~/spec_bench.log'`.
   NEVER block one ssh >300s.

## VERDICT (pending the host bench)
Local correctness is GREEN (lossless + verify logits fixed). The structural
prerequisite (gemm_mma amortization) is confirmed on 0.6B but only wins at the
target's larger matrix scale. Expected outcome on 4B: spec/baseline > 1.0 if the
4B seq=(K+1) verify approaches seq=1 cost AND mean-accept stays ~1.6-1.9; ceiling
~mean-accept x. Old spike was 0.54x. Bench fills in the actual ratio.
