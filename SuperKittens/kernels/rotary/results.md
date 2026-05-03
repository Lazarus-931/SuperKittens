# Rotary Embeddings (RoPE)

## Benchmarks vs MLX (B=1, fp16)

| H | L | D | Metal (us) | MLX (us) | vs MLX | GB/s |
|---|---|---|------------|----------|--------|------|
| 8 | 512 | 64 | 24 | 562 | **22.7×** | 93 |
| 8 | 1024 | 64 | 39 | 907 | **20.6×** | 114 |
| 8 | 2048 | 64 | 72 | 1514 | **22.5×** | 124 |
| 8 | 512 | 128 | 39 | 920 | **23.5×** | 114 |
| 8 | 1024 | 128 | 71 | 1566 | **22.2×** | 126 |
| 8 | 2048 | 128 | 196 | 2874 | **14.2×** | 91 |

Metal: `rope_qk` — tiled RoPE, half4, SIMD-group. 91–126 GB/s.

## Run

```bash
cd kernels/rotary/baseline && python3 bench.py   # MLX
xcrun metal ... && ./build/rope_bench            # Metal
```
