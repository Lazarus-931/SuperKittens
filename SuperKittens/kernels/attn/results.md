# Attention Benchmarks

Fused multi-head attention for Apple Silicon. Row-per-SIMD online softmax with cooperative K/V tile loading.

## Benchmarks vs MLX (B=1, H=1, fp16)

### d=128, causal (mha_causal)

| L | Metal (us) | MLX (us) | vs MLX | l2_rel |
|---|------------|----------|--------|--------|
| 256 | 81 | — | — | 0.00019 |
| 512 | 274 | — | — | 0.00019 |
| 1024 | 1133 | — | — | 0.00020 |
| 2048 | 3822 | 1406 | 0.37× | 0.00020 |

## Kernels

| Kernel | Head dim | Type |
|--------|----------|------|
| `mha_causal` | d=128 | Causal MHA, half4 vectorized |
| `mha_noncausal` | d=128 | Non-causal MHA, half4 vectorized |
| `fa_causal_64` | d=64 | Flash attention causal |
| `fa_noncausal_64` | d=64 | Flash attention noncausal |

## Run

```bash
cd kernels/attn/baseline && python3 bench.py   # MLX
./build/fa_bench                                # Metal
```
