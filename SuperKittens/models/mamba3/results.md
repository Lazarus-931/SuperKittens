# Mamba-3 Benchmarks

L=128, D=128, DQ=64, DV=64, H=4, CS=32. M2 GPU. fp16.

## Benchmarks vs MLX

| | Metal | MLX | vs MLX |
|---|---|------|--------|
| **SSM (MIMO)** | **0.91 ms** | **1.08 ms** | **1.19×** |
| **Full block (MIMO)** | **1.1 ms** | **1.31 ms** | **1.19×** |
| SSM (SISO) | — | 1.04 ms | — |
| Full block (SISO) | — | 0.84 ms | — |

SISO: DQ=4, DV=4, H=4 (scalar per head, no rotary). MIMO: DQ=64, DV=64, H=4.
Metal SSM verified against float64 CPU reference (max_err=0.57).
MLX bench verified e2e — calls `mamba3_block` → `mamba3_ssm` in ssm.py.

## Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `mamba3_ssm` | mamba3_ssm.metal | Selective scan, simdgroup MMA |
| `pre_ssm` | pre_ssm.metal | Norm + rotary |
| `post_ssm` | post_ssm.metal | SiLU gate |

## Run

```bash
cd kernels/mamba/mamba3/baseline && python3 bench.py   # MLX e2e
```
