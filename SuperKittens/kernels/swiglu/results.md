# SwiGLU Benchmarks

| Kernel | Description | Bandwidth | Efficiency |
|--------|-------------|-----------|------------|
| `fused_swiglu` | Fused SiLU + element-wise multiply | 85 GB/s | 95% peak |

Half4 vectorized, SIMD-group, zero barriers.
MLX comparison pending.
