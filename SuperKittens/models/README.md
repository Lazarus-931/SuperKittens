# Model Support

X = inference script exists, runs end-to-end on SuperKittens.
* = planned / in-progress.

Training column tracks whether backward pass + optimizer kernels are available.

## Language Models

| Model | Params | Attention | Norm | FFN Act | Inference | Training |
|---|---|---|---|---|---|---|
| LLaMA 3.2 1B | 1.2B | GQA d=64 (FA) | RMSNorm | SwiGLU | * | * |
| LLaMA 3.2 3B | 3.2B | GQA d=128 (MHA) | RMSNorm | SwiGLU | * | * |
| Qwen2.5 0.5B | 0.5B | GQA d=64 (FA) | RMSNorm | SwiGLU | * | * |
| Qwen2.5 1.5B | 1.5B | GQA d=128 (MHA) | RMSNorm | SwiGLU | * | * |
| Gemma 2B | 2.0B | MHA d=256 | RMSNorm | GeGLU | * | * |
| SmolLM2 135M | 135M | MHA d=64 (FA) | RMSNorm | SwiGLU | * | * |
| SmolLM2 360M | 360M | MHA d=64 (FA) | RMSNorm | SwiGLU | * | * |
| SmolLM2 1.7B | 1.7B | GQA d=128 (MHA) | RMSNorm | SwiGLU | * | * |
| TinyLlama 1.1B | 1.1B | MHA d=64 (FA) | RMSNorm | SwiGLU | * | * |
| Phi-3 mini | 3.8B | MHA d=128 (MHA) | RMSNorm | SwiGLU | * | * |

## State Space Models

| Model | Params | SSM | Norm | Conv | Inference | Training |
|---|---|---|---|---|---|---|
| Mamba-130M | 130M | mamba2_ssd | RMSNorm | Conv1d | * | * |
| Mamba-370M | 370M | mamba2_ssd | RMSNorm | Conv1d | * | * |
| Mamba-790M | 790M | mamba2_ssd | RMSNorm | Conv1d | * | * |
| Mamba3-130M | 130M | mamba3_ssm | RMSNorm | Conv1d | * | * |
| Mamba3-370M | 370M | mamba3_ssm | RMSNorm | Conv1d | * | * |

## Kernel Coverage Required

| Kernel | Inference | Training | Status |
|---|---|---|---|
| GEMM fp16 | X | X | done |
| FA d=64 | X | | done |
| MHA d=128 | X | | done |
| MHA d=256 (generic) | X | | fallback path works |
| RoPE | X | X | done |
| RMSNorm | X | X | done |
| RMSNorm + residual | X | | done (fusion/) |
| SiLU / SwiGLU gate | X | X | done |
| GELU / GeGLU gate | X | X | done |
| Causal Conv1d | X | | kernel exists, unverified |
| mamba2_ssd | X | | done |
| mamba3_ssm | X | | done |
| mamba3_pre_ssm (norm+rotary) | X | | done |
| mamba3_post_ssm (gate) | X | | done |
| GEMM backward (dA/dB) | | X | not started |
| Gradient reduction | | X | not started |
| AdamW optimizer | | X | not started |
| Cross-entropy loss | X | X | not started |
| FP8 GEMM | X | X | M5 only, not started |
| bfloat MMA | X | X | M2+, not started |
