# Kernel Library

Each directory in `kernels/` is a self-contained GPU operation with:
- `.metal` — Metal shader source
- `.h` / `.c++` — host-side C++ dispatch
- `baseline/*.py` — MLX reference implementation
- Benchmarks live in `benchmark/` (not inside `kernels/`)

## activation
| Kernel | Description | Host name |
|---|---|---|
| GELU | x * Φ(x) via tanh approximation | `gelu` |
| SiLU | x / (1 + exp(-x)) | `silu` |
| ReLU | max(0, x) | `relu` |

128 threads, 4 SIMD groups, half4 vectorized. Zero barriers.

## attn
| Kernel | Description | Host name |
|---|---|---|
| FA d=64 | Flash Attention, 1024 threads, simd_sum dot | `fa_causal_64`, `fa_noncausal_64` |
| MHA d=128 | Row-per-SIMD online softmax, half4 fast path | `mha_causal`, `mha_noncausal` |
| MHA generic | Scalar fallback, any head_dim | `mha_causal`, `mha_noncausal` |

Auto-dispatch via `attn.h`: `head_dim == 64` → FA, else → MHA.

## conv
| Kernel | Description |
|---|---|
| conv1d | Causal depthwise convolution |
| conv1d_silu | Conv1d + SiLU fused |
| conv2d | Im2col + simdgroup GEMM (tiled) |
| conv3d | 3D convolution |

Kernel sizes hardcoded for 3×3 kernel, C_in=64, C_out=64.

## fusion
Fused kernel compositions — two or more ops in one launch.

| Kernel | Fusion |
|---|---|
| `rms_residual` | RMSNorm(x + residual) |
| `gemm_bias_act` | GEMM + bias + GELU/SiLU/ReLU |
| `rms_rope` | RMSNorm(Q) + RMSNorm(K) + RoPE |
| `gemm_res_norm` | GEMM + residual add + RMSNorm |
| `gated_mlp` | SiLU(gate) * up → down (3 GEMMs fused) |

## gemm
| Variant | Path |
|---|---|
| fp16 | `fp16/gemm.metal` |
| fp8 | `fp8/gemm.metal` (M5+) |

Host dispatch in `gemm_host.h`, implementation helpers in `gemm_impl.h`.

## mamba
State space model kernels.

### mamba2
| Kernel | Description |
|---|---|
| mamba2_ssd | Selective scan, 128 threads, half4 vectorized |
| conv1d_silu | Causal depthwise conv + SiLU |
| gate_norm | Gating + RMSNorm |

### mamba3
| Kernel | Description |
|---|---|
| mamba3_ssm | Selective scan with trap discretization + rotary |
| pre_ssm | Fused RMSNorm(Q/K/B) + RoPE |
| post_ssm | SiLU gate: `silu(z) * ssm_out` |

Block orchestration in `mamba3_block.h`.

## rotary
| Kernel | Description | Host name |
|---|---|---|
| RoPE | Rotary position embedding on Q and K | `rope_qk` |

256 threads, half4 vectorized, in-place. cos/sin precomputed on host.

## swiglu
| Kernel | Description |
|---|---|
| fused_swiglu | SiLU-gated activation: `silu(x_gate) * x_up` |

## utils
| Kernel | Description |
|---|---|
| layernorm | LayerNorm: (x - μ) / σ * γ + β |
| rmsnorm | RMSNorm: x * γ / rms(x) |
