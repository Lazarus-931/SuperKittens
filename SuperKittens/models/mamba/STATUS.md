# SK Mamba 1 port — status

Target: `state-spaces/mamba-2.8b-hf` end-to-end inference on Apple Silicon, validated argmax-equal against HF reference for prompt `"Hi"`, decode tok/s measured.

This directory is the **scaffold** for the Mamba 1 (selective SSM) family, kept separate from `SuperKittens/models/mamba2/` per the `dev_env.md` "don't touch other models' code" rule. The mamba2 SSD kernels are not signature-compatible with Mamba 1.

## What is in place

- `download.sh`: `mamba-2.8b` entry resolves to `state-spaces/mamba-2.8b-hf`.
- HF source: sparse-checkout added at `~/transformers/src/transformers/models/mamba/`. Reference forward at `modeling_mamba.py:270` (`slow_forward`).
- Validation harness scaffold:
  - `SuperKittens/temp/mamba_validate/dump_hf_mamba.py` — hooks `in_proj`, `conv1d`, `x_proj`, `dt_proj`, `out_proj` per `MambaMixer`; writes `hf_ref.npz`.
  - `SuperKittens/temp/mamba_validate/layer_diff.py` — compares HF vs SK npz per key.
- `weights.h` — documented HF safetensor name map and `Dims` / `LayerWeights` / `ModelWeights` structs. Note: state-spaces uses `backbone.layers.{L}.mixer.*`, not `model.layers...`.
- `launcher.h` — C ABI: `sk_mamba_create / forward / reset_cache / dump_layer / destroy`.
- `mamba.py`, `__init__.py` — Python wrapper that registers `"mamba-2.8b"` via `sk.register` once the lib is functional. Currently raises `NotImplementedError` rather than silently returning garbage.

## What is NOT in place (work remaining)

Roughly in dependency order:

1. **Metal kernels** under `SuperKittens/models/mamba/`:
   - `mamba_in_proj_bf16.metal` — GEMM `[B,L,H] @ [2E,H]^T → [B,L,2E]` then chunk into hidden, gate.
   - `mamba_conv1d_silu_bf16.metal` — depthwise causal Conv1d kernel size 4 + SiLU. The mamba2 `conv1d_silu.metal` is depthwise/causal with `K=4` and `silu` — its signature **may be reusable**; verify weight layout (HF stores `[E,1,K]`, mamba2 wants `[C,K]`).
   - `mamba_x_proj_bf16.metal` — GEMM `[B,L,E] @ [dt_rank+2N,E]^T → [B,L,dt_rank+2N]`, then split into `(dt, B, C)`.
   - `mamba_dt_proj_softplus_bf16.metal` — GEMM `[B,L,dt_rank] @ [E,dt_rank]^T + bias`, then `softplus`. dt_proj bias is fp32 in HF.
   - **`mamba_selective_scan_bf16.metal`** — the hard one. Per `modeling_mamba.py:321-353`:
       - `A = -exp(A_log)` (fp32, `[E,N]`)
       - `discrete_A = exp(A * Δ)`, `[B,E,L,N]`
       - `deltaB_u = Δ * B_t * x_t`, `[B,E,L,N]`
       - Sequential recurrence `h_t = A_t * h_{t-1} + ΔB_u_t`, `y_t = h_t @ C_t`. For Mamba 1 there is **no chunked associative scan** like Mamba 2 — either a serial scan along L (correct, fine for prefill at L=1..hundreds) or a parallel scan over L per `(E,N)` lane. Persistent SSM state shape `[B,E,N]` fp32.
   - `mamba_d_residual_gate_bf16.metal` — `scan_out + x * D[:,None]`, then `* silu(gate)`.
   - `mamba_out_proj_bf16.metal` — GEMM `[B,L,E] @ [H,E]^T → [B,L,H]`.
   - `mamba_block_residual_rmsnorm_bf16.metal` — pre-norm residual + RMSNorm (`MambaRMSNorm`, `modeling_mamba.py:381`).
   - `mamba_embed_lookup_bf16.metal`, `mamba_final_norm_bf16.metal`, `mamba_lm_head_bf16.metal` — can likely reuse `gemma` equivalents if signatures match, but copy rather than share to keep ownership clean.
   - `mamba_argmax_bf16.metal` — simple.
2. **`launcher.c++`** — implements C ABI: model load (safetensors → bf16 fused buffers via `models/load/safetensor.*`), forward orchestration, SSM/conv state caches per layer, dump-layer for the validation harness.
3. **`mamba_model.h`** — orchestrator mirroring `slow_forward` step order: embed → for L: (rmsnorm + in_proj → conv1d_silu → x_proj → split → dt_proj+softplus → selective_scan → +D*x → *silu(gate) → out_proj + residual) → final norm → lm_head → argmax.
4. **Wire into `build.sh`** — verify it picks up `models/mamba/*.metal` and `*.c++`. (Existing build.sh globs `models/`; should be automatic.)
5. **Run validation harness** on validation host:
   - `./download.sh mamba-2.8b`
   - `python SuperKittens/temp/mamba_validate/dump_hf_mamba.py --dtype fp32`
   - Compare with `layer_diff.py`. Iterate.
6. **Decode tok/s benchmark** using the SSM single-step path (`update_recurrent_state` branch in HF, `modeling_mamba.py:236-263`).

## Validation status

Not yet runnable. Argmax match: **N/A**. Decode tok/s: **N/A**.

## Hard blocker

No technical blocker; the remaining work is multi-day kernel implementation effort, primarily the selective-scan Metal kernel and the surrounding glue. The mamba2 SSD kernels in `SuperKittens/models/mamba2/` cannot be dropped in unchanged because:

- Mamba 2 uses chunked associative scan over `(n_groups, n_heads, head_dim)` with shared B, C per group.
- Mamba 1 has per-channel `B, C` of shape `[N]` (no head structure) and a strictly sequential dependency `h_t = A_t * h_{t-1} + ΔB_u_t` with `A_t ∈ R^{E×N}` input-dependent — that is a fundamentally different scan pattern.

The right next step is to author `mamba_selective_scan_bf16.metal` (serial scan first, parallel scan optimization after correctness), then build outward from there.
