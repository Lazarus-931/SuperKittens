# Mamba-2 Benchmarks

L=128, D=128, E=64, N=64, H=2. M2 GPU. fp16.

## Benchmarks vs MLX

| | Metal | MLX | vs MLX |
|---|---|------|--------|
| **SSM (selective_scan)** | **1.49 ms** | **6.05 ms** | **4.1×** |
| **Full block** | — | **6.22 ms** | — |

Metal SSM verified against float64 CPU reference.
MLX bench verified e2e — calls `mamba2_block` → `selective_scan` in ssm.py.
Full block includes: in_proj, conv1d+SiLU, dt, SSM, silu gate, out_proj+skip.

## Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `mamba2_ssd` | mamba2_ssd.metal | SSD selective scan |
| — | mamba2_block.h | Host dispatch header |

## Run

```bash
cd kernels/mamba/mamba2/baseline && python3 bench.py   # MLX e2e
```
