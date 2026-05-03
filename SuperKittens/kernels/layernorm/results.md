# LayerNorm / RMSNorm Benchmarks

## Benchmarks vs MLX

| Op | Shape | MLX (us) | MLX GB/s | Metal GB/s | vs MLX |
|----|-------|----------|----------|------------|--------|
| layernorm | 1024×1024 | 791 | 5.3 | 59 | 11.1× |
| layernorm | 2048×2048 | 1900 | 8.8 | 59 | 6.7× |
| layernorm | 4096×4096 | 7673 | 8.7 | 59 | 6.8× |
| rmsnorm | 1024×1024 | 400 | 10.5 | — | — |
| rmsnorm | 2048×2048 | 1039 | 16.1 | — | — |
| rmsnorm | 4096×4096 | 3147 | 21.3 | — | — |

Metal: SIMD-group, zero barriers, half4 vectorized.

## Kernels

| Kernel | Description |
|--------|-------------|
| `layernorm` | LayerNorm, half4 | 
| `layernorm_residual` | LayerNorm + residual add |
| `rmsnorm` | RMSNorm, half4 |
| `rmsnorm_residual` | RMSNorm + residual add |

## Run

```bash
cd kernels/layernorm && python3 bench.py    # MLX
./build/layernorm_bench <metallib>          # Metal
```
