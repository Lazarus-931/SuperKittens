# mha_causal chunked-forward correctness fix — STATUS

Branch `dev-mha-chunkfix` off `origin/main` (907a0fe). Worktree, NO push/PR.

Fixes the documented correctness bug from the spec-decode spike (`dev-spec-decode`
temp/spec_decode/STATUS.md): a multi-token forward at `current_pos > 0` (seq=K on
top of a non-empty KV cache — spec-decode verify, chunked prefill, prompt
continuation) returns WRONG interior logits. Pure prefill (`current_pos=0`) and
seq=1 decode are correct and must stay bit-exact.

## Root cause (kernels/attn/attn.metal, `fa_d128<Causal>` = `mha_causal`)

The TG-staged Br=2 kernel runs `Hg*Br` simdgroups per threadgroup, each owning a
different `q_row` → a different `total_lim = kv_len - seq + q_row + 1` (the row's
causal key count). `k_smem`/`v_smem` are threadgroup-SHARED, but the cooperative
load gated each smem slot on the **loading thread's own** `total_lim`
(`if (col < total_lim)` at the old L211). A given key column `col` is loaded by
exactly one thread, whose simdgroup (= q_row) is fixed by the modular layout
`i = lid; i += NT`. When that owner row's `total_lim` does not cover `col`, the
slot is zeroed — even though a SIBLING row (larger `total_lim`, same TG) needs it.

At `current_pos=0` the per-row `total_lim` staircase happens to line up with the
slot-ownership modulo so every needed key's owner covers it (prefill correct). At
`current_pos>0` the staircase shifts; some keys land on an owner whose total_lim
masks them out, zeroing them for the rows that DO need them.

Confirmed by probe (Hg=2, seq=4, cur=15, kv_len=19): key col=16 is owned by
simdgroup 0 (q_row=0, total_lim=16), so `16<16` is false → key 16 zeroed; q_row=1
(total_lim=17) needed it → attends 0..15 instead of 0..16. Symptom: odd query
rows (r=1) lose their last (diagonal) key. Matches the spike's "rows 1,3 wrong"
finding exactly. Secondary defect: divergent `full_tiles` across simdgroups made
per-simdgroup `threadgroup_barrier` counts non-uniform (UB; survived only by luck).

## The fix (attn.metal, fa_d128, ~L175-247)

Drive the tile walk by a threadgroup-UNIFORM key extent `tg_lim` = `total_lim` of
the block's LAST active row (highest q_pos, clamped to seq-1), instead of the
per-thread `total_lim`. Smem load now gates on `col < tg_lim` so every key any
row in the block needs is loaded. Re-impose each row's own causal cutoff in the
score accumulation: a key column `col >= total_lim` contributes score `-INFINITY`
(→ exp=0 → zero weight, m/alpha unchanged), so masked steps are exact no-ops.
`mha_noncausal` (`Causal=false`) has `total_lim == tg_lim == kv_len` for all rows
→ unchanged.

WHY bit-exact on working paths: the driving row (total_lim==tg_lim) walks the same
keys in the same order with no masking; earlier rows see extra columns but each is
a true no-op (`max(m,-INF)=m`, `exp(0)=1`). Decode seq=1: tg_last_row clamps to 0
so tg_lim == the active row's total_lim; the inactive r=1 row is never written.

## Validation (local M2, build/libsk.metallib)

Kernel-level synthetic Q/K/V vs numpy causal SDPA (test_mha_chunk.py):
- BEFORE: chunk seq=4 @15/@17 max_rel 0.30-0.44 (rows 1,3 wrong). prefill/decode OK.
- AFTER : every case max_rel ≤ 2.8e-4 (fp16 noise). WORST over all cases 2.7e-4.

Bit-exactness vs pre-fix baseline metallib (test_bitexact.py):
- prefill current_pos=0 (seq 8/17/64/100): BIT-EXACT (byte-identical) for Hg=2,4.
- decode seq=1 (kv 9/18/64/201):           BIT-EXACT for Hg=2,4.
- chunk seq=4 @15/@17: CHANGED (expected — baseline was wrong).
- mha_noncausal (seq 8/17/70, chunk 4@70): BIT-EXACT for Hg=2,4.

Robustness sweep (test_mha_chunk-style, 41272 head/case checks):
current_pos 0..198 × seq {2,3,4,5,7,8,16} × Hg {2,4,5}: 0 failures, worst 2.8e-4.

End-to-end through real qwen3-0.6B-Q8_0 stack (test_e2e_capture.py, sk_qwen
capture hook): chunk-forward post-layer residual at interior positions vs full
prefill residual — BIT-IDENTICAL (max_rel 0.0) AFTER fix, and the production
prefill→greedy-decode stream is deterministic + coherent + in-vocab (regression
guard). NOTE: the e2e capture reads 0.0 for the PRE-FIX baseline too on these
token ids — the bug is data-dependent (the dropped diagonal key's softmax weight
is often small, and the residual stream dilutes the attn_out error), exactly the
spike's "argmax sometimes survives the bad attention / intermittent" finding. The
kernel-level synthetic test (which isolates attn_out directly) is therefore the
authoritative correctness probe; it deterministically triggers the bug (0.3-0.44
pre-fix) at the exact qwen3-0.6B attention dims (n_heads=16, n_kv=8, hd=128,
Hg=2). The e2e capture confirms NO regression to the model stack.

## Risk to production path

None observed. seq=1 decode and current_pos=0 prefill are byte-identical to the
pre-fix kernel (proven, not just within-tolerance). The change adds a few
comparisons per key in the score loop; bench on a real host before promotion, but
the decode/prefill numerics are unchanged.

## Files
- kernels/attn/attn.metal — fa_d128 (mha_causal) tile-walk + per-row mask fix.
- temp/mha_chunkfix/test_mha_chunk.py  — synthetic-vs-numpy correctness.
- temp/mha_chunkfix/test_bitexact.py   — baseline-vs-fixed byte diff.
- temp/mha_chunkfix/probe.py           — diagnostic (which keys attended).
- temp/mha_chunkfix/test_e2e_capture.py — e2e chunk-vs-prefill residual (real 0.6B).
