# Conv Benchmarks

1×56×56×64 input, 3×3 kernel, 64 output channels (valid, stride=1).
Output: 1×54×54×64. 215 MFLOPs. Apple M2.

## Benchmarks vs MLX

| Impl | Time (us) | GFLOPS | vs MLX |
|------|-----------|--------|--------|
| MLX conv2d | 658 | 327 | 1.00× |
| **Metal tiled simdgroup** | **886** | **243** | **0.74×** |

TM=32 (4×8 spatial), TN=64, BK=64. 64 threads, 2 SIMD groups. 16KB threadgroup.
18 barriers per threadgroup, 98 threadgroups. Stable, correct.

## Conv1D Benchmarks

B=1, L=128, C=256, K=4. Causal depthwise Conv1D.

| Impl | Time (us) | vs MLX |
|------|-----------|--------|
| MLX conv1d | 321 | 1.00× |
| **Metal conv1d** | **15** | **21×** |

SIMD-group per position, half4 vectorized, zero barriers. Weight [C, K].

## Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `conv2d_tiled` | conv2d.metal | Tiled simdgroup matmul |
| `conv3d` | conv3d.metal | 3D convolution, scalar |
| `conv1d` | conv1d.metal | Depthwise causal Conv1D, half4 |

## Run

```bash
cd kernels/conv/baseline && python3 -c "from conv2d import bench_conv2d; bench_conv2d()"   # MLX
./build/conv2d_bench build/conv2d.metallib                                                   # Metal
```

