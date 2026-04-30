# GEMM Checkpoint Results

Date: 2026-04-26

Shape: `1024 x 1024 x 1024`

Method:
- Metal timings are GPU median time from the GEMM smoke tests with `100` timed iterations.
- MLX timings are median wall-clock time from the GEMM baseline scripts with `100` timed iterations.
- MLX inputs are generated once and cached outside the timed region.
- Accuracy is reported only for Metal, against the CPU reference used by the smoke tests.

## Official Chart

| Path | Impl | Median ms | Relative To MLX | Accuracy vs CPU |
|---|---|---:|---:|---|
| Plain GEMM | Metal | 1.508 | 1.10x | `l2_rel=0.00021`, `max_abs=0.00389` |
| Plain GEMM | MLX | 1.662 | 1.00x | n/a |
| Fused GEMM + bias + SiLU | Metal | 2.101 | 0.82x | `l2_rel=0.00031`, `max_abs=0.00585` |
| Fused GEMM + bias + SiLU | MLX | 1.724 | 1.00x | n/a |

Notes:
- Plain GEMM now beats the cleaned MLX baseline on the target shape.
- Best observed plain Metal run in this tuning pass was `1.408 ms`; repeated reruns landed in the `1.47-1.51 ms` range.
- Fused GEMM is still behind MLX and is not the current optimization target.

## Commands

```bash
/tmp/gemm_smoke_test /tmp/gemm_fp16_1.metallib 1024 1024 1024 --iters 100
/tmp/gemm_bias_silu_smoke_test /tmp/gemm_fp16_2.metallib 1024 1024 1024 --iters 100
uv run python SuperKittens/kernels/gemm/baseline/mlx/bench.py 1024 1024 1024 --iters 100 --csv
uv run python SuperKittens/kernels/gemm/baseline/mlx/bench_bias_silu.py 1024 1024 1024 --iters 100 --csv
```
