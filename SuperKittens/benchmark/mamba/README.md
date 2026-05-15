# Mamba Benchmarks

Thin runners for Mamba kernel benches. Canonical Metal kernels live under
`SuperKittens/models/ssm/mamba2/` and `SuperKittens/models/ssm/mamba3/`.

## mamba2/
- `mamba2_smoke_test.cpp` — CPU-vs-GPU smoke bench for the Metal Mamba-2 forward kernel.
- `mamba2_parallel_bench.cpp` — parallel-scan SSD bench.
- `mamba2_siso_bwd_smoke_test.cpp` — backward smoke bench.
- `mamba2_mlx.py` — MLX baseline sweep.

## mamba3/
- `mamba3_smoke_test.cpp` — CPU-vs-GPU smoke bench for SISO/MIMO forward kernels.
- `mamba3_siso_bwd_smoke_test.cpp` — backward smoke bench.
- `mamba3_mlx.py` — MLX baseline sweep.
- `mamba3_siso_bwd_mlx.py` — MLX backward baseline.

## Top-level
- `run_smoke_sweeps.py` — convenience runner for Mamba-3 SISO/MIMO smoke sweeps.
