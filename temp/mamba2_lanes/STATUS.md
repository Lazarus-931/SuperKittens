# mamba2-130m batched-lane prefill (`sk_mamba2_prefill_batched`)

Branch: `dev-sk-mamba-lanes` (off `origin/main` @6931da9). Bench host: derek
(M4 base, 16 GB, CommandLineTools-only — dylib built with clang++, kernels
runtime-compiled via `SK_METAL_SRC_FALLBACK`).

## Problem
N-lane serving prefilled lanes SEQUENTIALLY: `sk_mamba2_prefill_lane` runs one
batch=1, seq=T forward per lane (`waitUntilCompleted` each), so N lanes pay N
full forwards — weights read N times, GPU underfilled (SSD scan at batch=1 is
only `H=24` threadgroups), and each forward wastefully runs the LM head over
all T rows when only the last row feeds the first token.

## Design
New additive ABI:
```c
int sk_mamba2_prefill_batched(h, ids /*[n_req*seq] lane-major*/, seq, n_req,
                              out_ids /*[n_req]*/);
```
ONE forward at `mp.batch=n_req, mp.seq=seq` (`ModelParams.lanes_last=n_req`).
Equal-length prompts only (documented scope; serving bench uses N distinct
equal-length prompts). Caller resets lanes first (Python wrapper
`Mamba2Model.prefill_batched` does it; `generate_batched(batched_prefill=True)`
wires it into the serving path).

## What was actually broken for batch>1 prefill (audit result)
Almost nothing in the kernels — lanes ARE the batch dim everywhere:
- `conv1d_silu` indexes `x[b*L*C + ...]` with the causal window clamped to
  `sp >= 0` inside the lane's own L block → NO cross-lane conv bleed (the
  feared lane-edge bug does not exist; verified bit-identical states).
- `conv_state_capture`, `mamba2_ssd` (`ssm_state[(b*H+h)*P*N]`), `gate_norm`,
  `rmsnorm`, splits, GEMMs (flat `T = batch*seq` rows): all batch-correct.

What WAS missing (all host-side dispatch):
1. No ABI/dispatch mode running prefill at batch=N with state offsets 0
   (kernels index lane b directly; `prefill_lane`'s per-lane byte offsets are
   batch=1-only).
2. Per-lane LAST-row logits/argmax: existing prefill argmaxes only global row
   T-1 (= last lane's last row). Fixed via `lanes_last` mode.
3. LM head over `batch*seq` rows would be a huge waste (T=1024 rows × V=50288).
   Added a blit gather of each lane's last `x_norm` row ((r+1)*seq-1) into a
   tiny `lanes_x (batch, D)` buffer → LM head at M=N → per-lane argmax (crib of
   `decode_all_rows`). Note: the sequential baseline pays the full T-row head
   per lane (existing behavior, untouched); part of the batched win is this
   last-rows-only head, which could later be retrofit to `prefill_lane` too.

Also found (pre-existing, untouched): `sk_mamba2_forward` sets
`mp.batch = cfg.batch`, so plain `forward()` on a batch=N handle is not
single-stream and reads `cfg.batch*seq` ids from the caller (OOB for a
seq-length array). Single-stream use requires a batch=1 handle, as before.

## Files changed (all additive; defaults keep old paths byte-identical)
- `SuperKittens/models/ssm/mamba2/mamba2_model.h` — `ModelParams.lanes_last`,
  `ModelBuffers.lanes_x`, last-row blit gather + M=N LM head + per-lane argmax.
- `SuperKittens/models/ssm/mamba2/launcher.c++` — `sk_mamba2_prefill_batched`,
  `lanes_x` alloc/release.
- `SuperKittens/models/ssm/mamba2/launcher.h` — ABI declaration.
- `SuperKittens/models/ssm/mamba2/mamba2.py` — `prefill_batched()` wrapper +
  `generate_batched(..., batched_prefill=True)`.

## Gate results (derek, M4 base)
### Gate 1 — lane isolation + correctness: PASS
N=8 distinct prompts (T=128, seeded rng ids): batched-prefill first tokens ==
sequential-prefill first tokens; greedy continuation via
`sk_mamba2_decode_batched` token-for-token identical for 33 tokens, 8/8 lanes;
per-lane final conv_state + ssm_state rel-L2 = 0.0 (bit-identical — same
kernels, same per-lane order); batched logits finite (absmax 81.8).

### Gate 2 — existing paths byte-identical: PASS
Same-host A/B of dylibs built from pristine main vs this branch: single-stream
(batch=1 handle) 24-token greedy generation → tokens identical, last-row
logits byte-identical. Sequential `prefill_lane` + `decode_batched` path is
untouched code (and is the A side of gate 1/3, running correctly).

### Gate 3 — e2e serving TTFT (A = N× prefill_lane, B = one prefill_batched)
Protocol: one process, one batch=8 handle; lanes reset outside the timed
region; 2 warmups + 7 reps, A/B alternated per rep, 0.3 s gaps, median.

| T | N | A seq (ms) | B batched (ms) | speedup |
|-----|---|--------|--------|-----------|
| 128 | 2 | 84.74  | 70.21  | **1.21×** |
| 128 | 4 | 164.11 | 125.29 | **1.31×** |
| 128 | 8 | 322.15 | 237.79 | **1.35×** |
| 256 | 2 | 154.32 | 127.06 | **1.21×** |
| 256 | 4 | 306.37 | 237.49 | **1.29×** |
| 256 | 8 | 605.24*| 459.95 | **1.32×** |

(*A from the same-protocol pristine-main run, 606.04 on the new dylib —
identical within noise.) PASS: ≥8% everywhere (21–35%). Sanity: A scales
linearly in N (84.7 → 164.1 → 322.2 ms ≈ 2× per doubling), confirming the
baseline really ran N forwards; A/N at T=128 ≈ 40–42 ms matches the prior
~45 ms single-stream TTFT profile. No multi-× — at T=128–256 the batch=1
GEMMs (M=128/256 rows) already occupy the M4 reasonably, so the batched win
is weight-read amortization + last-rows-only LM head + per-forward overhead,
while the row-proportional body work is unchanged.

### Gate 4 — decode_batched unchanged: PASS
Aggregate decode tok/s (64 lockstep steps), same process/protocol:

| N | main dylib | new, after A | new, after B |
|---|-----------|--------------|--------------|
| 2 | 186.0–188.1 | 187.8–188.1 | 187.4–187.8 |
| 4 | 345.6–346.6 | 348.8–349.0 | 346.7–349.3 |
| 8 | 592.5 | 591.6–592.8 | 589.8–591.1 |

All within ±1% (gate: ±2%). Sequential prefill_lane times also identical
across dylibs (322.19 vs 322.15 ms @ T=128 N=8).

## Verdict
POS. Batched-lane prefill lands 1.21–1.35× e2e serving TTFT (N=2–8, T=128/256,
M4 base) with token-for-token / bit-identical-state correctness vs the
sequential path, zero change to existing paths (byte-identical single-stream
logits, decode_batched within ±1%). Scope: equal-length prompts only.

## Repro (derek)
- `~/sk-mamba-lanes`: rsynced worktree; dylib via `build_both.sh`
  (`libsk_base.dylib` = pristine main, `libsk.dylib` = this branch).
- Gates: `bash temp/mamba2_lanes/run_lab.sh temp/mamba2_lanes/gate_correctness.py`
  (gate1), `run_gate2.sh` (gate2), `bench_ttft.py` under new + base dylib
  (gates 3/4). Logs: `temp/mamba2_lanes/{gate1,gate2,bench_new,bench_base}.log`.
