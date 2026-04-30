# Mamba Benchmarks

This directory collects Mamba benchmarking entrypoints in one place.

Files:
- `mamba2_smoke_test.cpp`: CPU-vs-GPU smoke benchmark for the Metal Mamba-2 forward kernel.
- `mamba2_mlx.py`: MLX baseline sweep for Mamba-2.
- `mamba3_smoke_test.cpp`: CPU-vs-GPU smoke benchmark for the Metal Mamba-3 SISO/MIMO forward kernels.
- `mamba3_mlx.py`: MLX baseline sweep for Mamba-3 helpers.
- `run_smoke_sweeps.py`: convenience runner for Mamba-3 SISO/MIMO smoke sweeps.

Notes:
- The canonical Metal kernels still live under `SuperKittens/kernels/mamba/...`.
- These benchmark entrypoints are thin runners so benchmarking is organized separately from kernel code.
