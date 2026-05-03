# GEMM Benchmarks

FP16 GEMM on Apple M2. Metal vs MLX.

## Plain GEMM — 1024×1024×1024

| Impl | Median ms | vs MLX | Accuracy |
|------|-----------|--------|----------|
| Metal | 1.508 | 1.10× | l2_rel=0.00021, max_abs=0.00389 |
| MLX | 1.662 | 1.00× | — |

## Fused GEMM + Bias + SiLU — 1024×1024×1024

| Impl | Median ms | vs MLX | Accuracy |
|------|-----------|--------|----------|
| Metal | 2.101 | 0.82× | l2_rel=0.00031, max_abs=0.00585 |
| MLX | 1.724 | 1.00× | — |

Kernel: `gemm_fp16` (unified NN/NT/TN, simdgroup MMA, 64 threads, BM=32 BN=64 BK=32).

Fixed-shape specializations: 2048×3072×4096, 3072×2048×4096, 4096×4096×4096.
