# Kernel Index

Each kernel dir under `SuperKittens/kernels/` ships `.metal` source, `.h`/`.c++`
host dispatch, and (where relevant) MLX baselines under `baseline/`.

Canonical perf numbers: `best.md` (repo root).

## kernels/

| Dir | Contents |
|---|---|
| `activation` | GELU, SiLU, ReLU (half4, no barriers) |
| `attn` | MHA d=64 / d=128 (prefill, Br=4 queries/TG) |
| `flash_attn` | ds4-derived flash attn, dk/dv up to 512 (decode, MLA) |
| `paged_attn` | Paged KV-cache attention |
| `conv` | conv1d, conv1d_silu, conv2d, conv3d |
| `fusion` | rms_rope, rms_residual, add_rmsnorm, gemm_bias_act, gemm_res_norm, gated_mlp (+ bf16/geglu variants), gemv_bf16_m1, gemv_swiglu_m1, kv_up_pair, silu_mul |
| `gemm` | fp16 (`fp16/gemm.metal`), fp8 (`fp8/gemm.metal`, M5+), M=1 quant matvecs (q8_0/q4k/q6k/q2k/iq2xxs), batched seq>1 MMA GEMM (`gemm_mma.metal`: f16/q8_0/q4k) |
| `moe` | router, router_v3, down_scatter, swiglu_pair (+ Q2K / IQ2XXS quantized) |
| `ops` | add, cast, causal_mask, kv_cache, sample, split, transpose |
| `rotary` | RoPE on Q/K (standalone path; usually fused via `fusion/rms_rope`) |
| `swiglu` | fused_swiglu: `silu(gate) * up` |
| `utils` | rmsnorm, layernorm, embedding |

Auto-dispatch: `attn.h` routes `head_dim==64` → FA64, else MHA.

## models/

Per-model orchestration (weights loader, launcher C ABI, Python ctypes wrapper,
model-specific kernels):

| Dir | Model family |
|---|---|
| `qwen` | Qwen3 (0.6B canonical, ICB decode) |
| `gemma` | Gemma 2/3/4 (incl. gemma4 SWA attn at d=256/512) |
| `deepseek` | DeepSeek (MLA via flash_attn dk=dv=512) |
| `ssm/mamba2` | Mamba2 SSD (see `ssm/mamba2/STATUS.md`) |
| `ssm/mamba3` | Mamba3 SSM (pre_ssm RMSNorm+RoPE, mamba3_ssm, post_ssm gate) |
| `load` | Shared HF safetensors loader |
