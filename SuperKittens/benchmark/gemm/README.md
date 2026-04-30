# GEMM Benchmarks

This directory holds standalone GEMM validation and benchmarking entrypoints.

Current files:
- `gemm_smoke_test.cpp`: CPU-vs-GPU correctness check for the FP16 NN GEMM scaffold.
- `gemm_bias_silu_smoke_test.cpp`: CPU-vs-GPU correctness check for the first fused GEMM epilogue.
- `gemm_mlx.py`: MLX performance sweep for plain GEMM.
- `gemm_bias_silu_mlx.py`: MLX performance sweep for fused bias+SiLU GEMM.

Notes:
- The current FP16 GEMM path is a stable external interface first, not the final fast path.
- The first fused epilogue is `bias + SiLU` on top of FP16 NN GEMM.
- Fixed-shape entry points already exist for `2048x3072x4096`, `3072x2048x4096`, and `4096x4096x4096`.
- Future TK-style optimized kernels should keep the same `GemmParams` contract and kernel names where possible.
- Both smoke tests now support `--dump <dir>` so MLX can validate against the exact same inputs and GPU outputs.
- Both smoke tests support `--no-verify` for large perf-only runs and `--iters N` to control timing depth.
- Plain and fused smoke tests also support `--force-generic` so specialization perf can be compared against the generic kernel on the same shape.
- MLX baseline scripts live under `SuperKittens/kernels/gemm/baseline/mlx/`:
  - `bench.py` and `bench_bias_silu.py` for perf
  - `accuracy.py` for dump-based accuracy checks
- MLX GEMM perf scripts cache their inputs outside the timed region. Do not compare against older numbers gathered before that cleanup.
- The current checkpoint table lives in `RESULTS.md`.
