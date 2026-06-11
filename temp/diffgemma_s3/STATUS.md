# DiffusionGemma Stage 3a — single-host Metal forward perf

Continues temp/diffgemma_s2/STATUS.md. Baseline (Stage-2 gate runs, amelia
M4 16 GB, C=256, S=16, seed 1234, SC on): **49.5-50.4 s/step wall, forward
98% of wall**. CPU llama.cpp reference on the same box: 29.85 s/step.
Targets in order: (1) beat the CPU reference, (2) approach the paging-bound
floor. Lab: amelia `~/sk-diffg-s3` (s2b artifacts read-only).

## Per-stage timing breakdown — BEFORE (instrumented Stage-2 code)

`forward_metal.Timing` (DG_TIMING=1), tools/profile_fw.py: p1 prompt (P=23),
mask canvas, cold + warm zero-SC + warm SC forward. Warm zero-SC 44.9 s,
SC 47.6 s (matches the Stage-2 fleet numbers). Dominant stages (warm, s):

| stage | wall_s | GB moved | share |
|---|---|---|---|
| wb:ffn_gate_up_exps (memcpy stream) | 21.0 | 8.56 | 47% |
| wb:ffn_down_exps (memcpy stream) | 16.0 | 6.57 | 36% |
| wb: all other weights (memcpy stream) | ~2.5 | 1.0 | 6% |
| wb:sc_embT (SC forward only) | 3.3 | 1.48 | 7% of SC |
| gpu: all GEMMs (moe 1.4, proj 0.7, attn 0.2, head 0.3, sc 0.5) | ~2.6 | — | 6% |
| host glue (moe pack/geglu/scatter 0.9, norms/rope 0.2, gc 0.35) | ~1.7 | — | 4% |
| h2d/d2h activations | 0.15 | 1.6 | <1% |

**83% of the forward is the per-step weight memcpy stream at 0.41 GB/s
effective** — the Stage-1/2 WeightBufs slots memcpy the full 15.1 GB of
expert+dense weights out of the mmap'd GGUF every forward, and the
madvise(MADV_DONTNEED) issued after each fill (the Stage-1 swap-storm fix)
forces the whole model to be re-read FROM DISK every step through
single-threaded Python page faults. GPU compute is ~2.6 s/forward — the
MoE expert GEMM loop itself (the suspected lever-1) is NOT the cost; its
*weight delivery* is.

## Lever table (per-forward, profile_fw.py p1 P=23; e2e A/B at bottom)

| lever | warm zero-SC (s) | warm SC (s) | gate |
|---|---|---|---|
| stream (Stage-2 prod, baseline) | 44.9 | 47.6 | — |
| mapped (no-copy MAP_SHARED) | 44.45 | 42.97 | bit-identical both legs |
| resident 4 GB (3.91 pinned) | 40.97 | 42.81 | bit-identical both legs |
| resident 6 GB (5.94 pinned) | 44.49 REGRESS | 45.16 REGRESS | swap 1.8->2.3G climbing |
| + pread streaming (lever B, 4 GB) | **15.58** | **16.55** | bit-identical both legs; swap flat 1.73G |
| pread + 6 GB re-probe | 15.48 | 17.71 | swap 1.7->2.85G; pins compressed — NO WIN, budget stays 4 GB |
| pread + 3 threads (lever C probe) | (pending) | (pending) | (pending) |

Resident 6 GB regression anatomy: pinned MTLBuffers are anonymous memory,
not wired — at 5.94 GB pinned next to the ~4 GB colima VM the compressor
took ~2.5 GB of them mid-forward (RSS 5.9 -> 3.4 GB, swap climbing), so the
GPU re-faulted pins from swap instead of weights from the GGUF. Practical
pin ceiling at mmap-stream pressure: ~4 GB budget. Re-test 5-6 GB after
lever B (F_NOCACHE removes the page-cache pressure squeezing the pins).

## Levers

### Lever 1 — no-copy MAP_SHARED weight binding (MappedWeights): BREAK-EVEN

llama.cpp-style no-copy binding: MAP_SHARED + PROT_READ libc mmap regions of
the GGUF wrapped into newBufferWithBytesNoCopy MTLBuffers; sc_embT sidecar
the same (kills the 8x184 MB chunk stream). Bit-identical to stream, but
**measured break-even** (44.45 vs 44.9 warm zero-SC; logs/mapped1.log): the
15.6 GB working set cannot stay file-cache-resident on the 16 GB host, so
the pager re-faults nearly the whole model from SSD every forward — the wall
just moved from wb: memcpy stages into the gpu: waitUntilCompleted stages
(GPU-access faulting, ~0.36 GB/s effective). The pager cannot be the
residency layer when the working set exceeds RAM. Kept as DG_WEIGHTS=mapped.

### Lever A — budgeted resident weight cache (ResidentWeights): LANDED 537aef0

DG_WEIGHTS=resident (new default). Backbone (embed/head, attn, dense ffn,
SC projections — 1.61 GB, 209 tensors) copied once into permanent
MTLBuffers; expert layers pinned fixed-prefix (layer order) while total
resident <= SK_DG_RESIDENT_GB; the remainder streams through the Stage-2
WeightBufs slots. Pinned anonymous copies are not evictable as clean file
cache — per-step disk traffic capped at (model - budget) bytes.

At 4.0 GB (3.91 pinned = backbone + 4.5 expert layers), p1 P=23:
- warm zero-SC **40.97 s** (stream 44.9, -8.8%), SC **42.81 s** (47.6, -10.1%)
- logits BIT-IDENTICAL to stream on both legs (gen/resident4_*.f32)
- swap flat 1.51 G, EXIT 0
- backbone stages collapse: attn_q 8.01->0.14 s, ffn_up 7.76->0.07 s,
  token_embd 5.58->0.36 s, SC subgraph 2.64->1.45 s
- remaining wall: 12.83 GB expert streaming at **0.363 GB/s** = 35.3 s,
  88% of the forward -> streaming RATE is the next lever, not residency
  share (each pinned GB only saves ~2.8 s at this rate)

### Lever B — F_NOCACHE pread streaming (in flight)

The 0.363 GB/s wb: rate is single-threaded page-fault servicing through the
mmap slice (16 KB faults + DONTNEED defeating readahead), not SSD bandwidth.
Replace with os.preadv straight into the slot MTLBuffer on an F_NOCACHE fd
(streamed bytes are re-read every step; caching them only pressures the
pinned buffers + colima). DG_STREAM_IO=mmap keeps the old path.
Gate: bit-identity (same bytes), then same profile A/B at the chosen budget.
