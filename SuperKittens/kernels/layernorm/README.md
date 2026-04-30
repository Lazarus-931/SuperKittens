# LayerNorm

SIMD-group layernorm. Each SIMD group processes one row independently — no barriers, no threadgroup memory.

## API

```
layernorm(x, gamma, beta, y, rows, d, eps)  // 7 buffers
```

Grid: `(1, ceil_div(rows, 4), 1)`, 128 threads.

## Algorithm

Single-pass sum/sumSq → `simd_sum` reduction within each SIMD group → normalize + affine.
```
mean = sum(x)/d
var  = sum(x^2)/d - mean^2
y    = (x - mean) * rsqrt(var + eps) * gamma + beta
```
