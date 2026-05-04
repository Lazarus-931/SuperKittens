# SuperKittens — benchmark times (Apple M2)

## Mamba-2 block (L=128, D=128, E=256, H=4, N=64)

| Kernel | Time | vs MLX |
|--------|------|--------|
| Full block (5 kernels) | 2.51 ms | 2.23× |
| SSM only | 1.49 ms | 7.9× |
| conv1d + silu | 40 us | — |
| gate + norm | <1 us | — |
| in_proj + out_proj | ~1 ms | — |

## Attention (causal MHA, d=128)

| seq | Time | vs MLX |
|-----|------|--------|
| 256 | 250 us | 1.45× |
| 2048 | 3.8 ms | 0.71× |

## Norms (d=768)

| Kernel | Time | vs MLX |
|--------|------|--------|
| LayerNorm | 16 us | 21× |
| RMSNorm | 16 us | — |
| LN + residual (fused) | 19 us | 1.6× |

## GEMM (fp16, NN, 1024³)

| | Time |
|---|---|
| Metal | 1.81 ms |
| MLX | 1.17 ms |

## Element-wise

| Kernel | Throughput |
|--------|-----------|
| SwiGLU | 85 GB/s (95% peak) |
| GELU / SiLU / ReLU | ~80 GB/s |
