# Gemma 4 architecture spec (E2B / E4B)

Derived from HF `transformers/models/gemma4/{configuration_gemma4.py,modeling_gemma4.py}`.

## E2B real config (per `text_config` of `google/gemma-4-E2B-it`)

| field | value | notes |
|---|---|---|
| n_layers | 35 | |
| d_model (hidden_size) | 1536 | |
| n_heads | 8 | |
| n_kv_heads | 1 | |
| head_dim (sliding) | 256 | |
| global_head_dim (full) | 512 | per-layer-type |
| intermediate_size (n_int) | 6144 | base; 2× for kv-shared layers |
| use_double_wide_mlp | True | layers ≥ (n_layers − num_kv_shared_layers) get n_int × 2 |
| num_kv_shared_layers | 20 | layers L15..L34 share KV from prior |
| hidden_size_per_layer_input (PLE_dim) | 256 | PLE table is `[vocab, n_layers * 256]` |
| vocab_size | 262144 | |
| vocab_size_per_layer_input | 262144 | |
| sliding_window | 512 | for sliding_attention layers |
| final_logit_softcapping | 30.0 | `logits = tanh(logits/cap) * cap` after LM head |
| rms_norm_eps | 1e-6 | |
| rope_theta_global | 1_000_000 | full_attention layers |
| rope_theta_local | 10_000 | sliding_attention layers |
| partial_rotary_factor (full only) | 0.25 | only 25% of head_dim is rotated for full_attention |

## Per-layer features

Layer type comes from `text_config.layer_types[L]` (explicit array). For E2B: L4, L9, L14, L19, L24, L29, L34 are `full_attention`; rest are `sliding_attention`. **Last layer must be `full_attention`.** Default fallback pattern is `((L+1) % 6 == 0)` but real configs override.

1. **Per-layer head_dim**: sliding=256, full=512.
2. **Per-layer n_int**: layer L gets `intermediate_size × (2 if L >= n_layers - num_kv_shared_layers and use_double_wide_mlp else 1)`.
3. **KV sharing**: layers ≥ `first_kv_shared_layer_idx = n_layers - num_kv_shared_layers` (E2B: 15). These layers have NO `k_proj`, `v_proj`, `k_norm`, `v_norm` weights. At runtime they reuse K/V from `store_full_length_kv` layer — the last non-shared layer of the same `layer_type` before the boundary.
4. **V-norm**: Q and K get full rmsnorm with learned gamma. V gets rmsnorm WITHOUT gamma (`Gemma4RMSNorm(with_scale=False)`).
5. **Partial RoPE** for full_attention only: only the first `int(head_dim * 0.25)` dims of Q/K are rotated; remaining stay as-is.
6. **PLE injection** per layer: `per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm`, see HF `Gemma4TextDecoderLayer.forward`.

## Final logit softcap

After LM head GEMM produces logits of shape `(batch, seq, vocab)`:
```
logits = tanh(logits / cap) * cap     # cap = 30.0 for E2B/E4B
```

## SK port status (as of 2026-05-12)

✅ Kernels: attn (mha_causal d=128, plus gemma4-specific d=256 local / d=512 global variants in `models/gemma/gemma4/attn.metal`), rope, rmsnorm, gemm, gated_mlp.
✅ Loader infra: `WeightStore`, safetensor reader, BF16→FP16 cast.
✅ Top-level: `sk.load("gemma4-e2b")`, registry.
✅ Config reader: reads `text_config.head_dim`, `global_head_dim`, `hidden_size_per_layer_input` from `config.json`.

❌ **Per-layer n_int** (double-wide MLP): SK assumes uniform `n_int` across layers. Needs per-layer offsets in launcher allocation and `dispatch_layer`.
❌ **KV sharing** (`num_kv_shared_layers`): SK expects every layer to load k/v/k_norm/v_norm tensors. For L≥15 these tensors don't exist in the checkpoint.
❌ **V-norm**: not applied. Add a no-gamma rmsnorm step on V before attention.
❌ **Partial RoPE**: full-attention layers in SK currently rotate the whole head; need to rotate only `partial_rotary_factor * head_dim` dims.
❌ **Final logit softcap**: not applied. Trivial new kernel or fused into argmax.
❌ **AltUp**: skipped intentionally — single-stream simplification. Acceptable for v0.
❌ **Layer types array**: SK computes via `local_period`; should read `text_config.layer_types[]` directly.
❌ **HF comparison harness**: no per-layer diff against reference.

## Recommended porting order

1. HF reference activation dump (one-shot, lets every later step be verified).
2. Layer_types reader + per-layer n_int + KV sharing (together — they all reshape the weight loader).
3. V-norm + partial RoPE + softcap (kernel additions).
4. PLE pipeline validation (already wired structurally).
5. End-to-end logit comparison vs HF.

## L0.out diagnostic note (2026-05-13, derek)

Investigation status of the L0.out rel=0.020 residual divergence (validation against `hf_ref.npz` for prompt `[10979, 236888]`):

### Cleanly verified
- `W_per_layer_projection` is bit-exact vs HF safetensors after the loader transpose (`W_sk == W_hf.T`).
- `W_per_layer_input_gate`, `W_post_per_layer_input_norm`, the PLE-table embed scale (sqrt(ple_dim)=16), and the `(proj + token_id) * 1/sqrt(2)` context-mix all match HF `modeling_gemma4.py:1754-1787`.
- Gemma4RMSNorm in SK (`x * invrms * gamma`, *no* `1+w`) matches HF — Gemma4's RMSNorm differs from Gemma3/3n.
- Decoder layer formula `out = layer_scalar * (residual + post_per_layer_input_norm(...))` matches HF `modeling_gemma4.py:1428-1437`.
- End-to-end numpy reproduction with HF weights + SK's `pre_ple` reproduces `hf["L0.out"]` at max-abs=0.045.

### Residual bug
SK GPU output differs from the numpy reproduction by max-abs=0.46, concentrated on `post_per_layer_input_norm.weight == 5.4375` channels. Back-solving the kernel formula, SK's `ple_proj_back` differs from HF's `per_layer_projection` by ~4.8 on those channels — a structural miscompute somewhere in the PLE projection path (`gemma4_gemm_bf16_fp32_out` or its bf16 input from `gemma4_ple_gate_act`), not a precision/bf16 rounding issue.

### Next probe
- Add dump taps for `ple_gate_out` and `ple_proj_back` (only `pre_ple` is currently exposed).
- Bisect by feeding SK's `ple_gate_out` into the numpy probe (`temp/gemma4_validate/probe8.py`) to isolate which of the two GEMMs is the offending kernel.
- As a sanity check, swap `P.gemm_fp32_out` to `P.gemm` in `dispatch_ple_inject` step 3 to see if accuracy moves.

Probes: `temp/gemma4_validate/probe{5..11}.py` on derek (gitignored).

## L0.out bisection result (2026-05-13, derek, probe9/10/11)

After adding `L0.ple_gate_out`, `L0.ple_gated`, `L0.ple_proj_back` dump taps:

| Stage | Kernel | Result vs fp64 numpy reference |
|---|---|---|
| 1 — gate GEMM | `gemma4_gemm_bf16` | **OK** — max=0.0038, matches HF `L0.per_layer_input_gate` |
| 2 — gate act  | `gemma4_ple_gate_act` | **WRONG** — sk_gated vs `gelu(gate)*ple_slice_L0`: max=5.16, mean=0.13 |
| 3 — proj GEMM | `gemma4_gemm_bf16_fp32_out` | **OK given inputs** — `sk_proj == sk_gated @ W_proj.T` bit-exactly |

So the L0.out=0.46 error originates entirely in step 2 (`gemma4_ple_gate_act`).

Probe11 shows the effective per-element multiplier SK applies (= sk_gated / gelu(gate))
does not match `per_layer_inputs[L=0]` for any layer L=0..5 — closest is L=0 with
mean abs diff ~0.47, so it is **not** a layer-index off-by-one. Root cause is one of:

1. `gemma4_ple_context_mix` writes `per_layer_inputs` in a different layout than the
   `(T, n_layers, PLE_dim)` that `gemma4_ple_gate_act` assumes.
2. Stride/shape miscount in `gemma4_ple_gate_act` (`ple_off = (t*n_layers + L)*P`).

Next probe: add `L0.per_layer_inputs` dump tap so we can compare SK's actual ple-slice
against the HF/numpy reference directly.
