# Mamba 3 SK port — STATUS

Branch: `dev-sk-mamba3-hang-fix`.

## Long-sequence "hang" — RESOLVED (it was a numerics bug, not a hang)

The reported "mamba3 hangs at seqlen>1024" came from the **H100/CUDA** mamba3
bench (see user memory `mamba3_bench_h100.md`), a different codebase. The Metal
`mamba3_ssm` kernel in this tree **does not hang** at any sequence length —
verified L=512 … 32768 all complete (32K tokens in ~116 ms on this Mac, scaling
linearly in L). The chunk loop already tiles correctly (`nC = ceil(L/CS)`,
fixed per-chunk threadgroup memory ≈ 23 KB ≤ 32 KB), so there is no fixed buffer,
grid overflow, or untiled loop that breaks past 1024.

The real problem was that `mamba3_ssm` produced **wrong scores at all seqlens**
(rel error ~0.5–1.6 vs an exact per-chunk reference), which is what gates
correctness for long-sequence prefill.

### Root cause

The intra-chunk score matmul `Q @ K^T` (simdgroup MMA) was broken two ways:

1. **Partial sum read back through threadgroup memory.** The k-loop kept the
   `simdgroup_float8x8` accumulator in threadgroup `sc` and re-read it each
   k-iteration via `acc.thread_elements()[ii*8+jj] = sc[...]`. But
   `thread_elements()[]` does **not** map to logical (row,col); only the
   lane-scatter on store does. So every k-iteration past the first accumulated
   onto garbage. The bug was masked for `DQ == 8` (single k-iteration, reads the
   zeroed `sc`), which is why scalar/SISO shapes looked fine.

2. **One tile per simdgroup.** `if (simd < MR*MC)` assumed ≤ 4 output tiles, but
   `CS = 32` needs `(32/8)^2 = 16` tiles and a 128-thread TG has only 4
   simdgroups — 12 of 16 score tiles were never written.

### Fix (`mamba3_ssm.metal`)

- Accumulator stays in registers across the full `DQ/8` k-loop; the lane-scatter
  store to `sc` happens once after the loop (matches `kernels/gemm/fp16/gemm.metal`).
- Each simdgroup strides over the tiles it owns (`for tile = simd; tile < MR*MC; tile += NSG`).
- Tail chunks with `cl` not a multiple of 8: pad `Qc/Kc` rows to `clp = ceil8(cl)`
  with zeros, tile the MMA over `clp`, and stride `sc` by `clp` so padded
  rows/cols (zero scores) don't alias real entries. This fixes arbitrary
  seqlens (e.g. L=1025/1030/1044) that previously dropped tail rows/cols.

### Validation (local, this Mac, Metal)

Oracle: a pure-numpy reference (`temp/mamba3_hang/np_ref.py`) that mirrors the
kernel's exact per-chunk math (the fix touches only the score MMA + tail tiling,
not the SSM recurrence, so this is the correct oracle for "did I change
numerics elsewhere"). All `≤ 1e-3` relative:

| shape | seqlens (all PASS) |
|---|---|
| BH=4 DQ=64 DV=64 CS=32 | 16, 32, 64, 128, 512, 1024, 1025, 1030, 1044, 2048, 3000, 4096, 8192, 16384, 32768 |
| CS=16 | 512, 1025, 2048 |
| DV=128 (2 DV tiles) | 512, 2048 |
| DQ=32 | 512, 2048 |
| BH=8 | 2048 |

No hang, no NaN, linear scaling. The MMA writeback in isolation
(`temp/mamba3_hang/test_scores.py`) matches numpy `Q@K^T` for cl ∈ {8,16,32}.

## Note on the MLX baseline

`baseline/ssm.py` uses a **different formulation** than the kernel: it omits the
`b_scale` factor on `K^T @ V` and the cross-chunk `a_cs_off` term in the rotary
angle. np_ref vs the MLX baseline differ by rel ~1.2, so the MLX baseline is
**not** a valid oracle for this kernel as-is. Reconciling the two
formulations (decide which matches the real Mamba-3 spec and align both) is
follow-up work; it is orthogonal to the score-MMA fix here.
