# flash_attn (ds4-derived MLA flash attention)

Source: antirez/ds4 (`metal/flash_attn.metal`), MIT licensed. Function-constant
templated; instantiated at compile time for specific (dk, dv) pairs.

This is the **second** flash attention in SK, alongside `kernels/attn/`. The
two are kept side by side because they target different shapes:

| kernel | head_dim | tile | use case |
|---|---|---|---|
| `kernels/attn/mha_causal` (SK) | 64, 128 | Br=4 queries / TG | prefill, multi-query, d≤128 |
| `kernels/flash_attn/flash_attn_ext_vec` (ds4) | 128, 512 | Br=1 query / TG | decode, MLA at dk=dv=512 |
| `models/gemma/gemma4/attn.metal` | 256, 512 | Br=1 query / TG | gemma4 SWA |

## Bench: ds4 vs SK at d=128 (causal MHA, fp16, B=1)

GPU-timestamp median of 20 iters, M2.

| (H, S, D=128) | SK ms | ds4 ms | winner |
|---|---|---|---|
| 8, 128 | 0.12 | 0.16 | SK 1.3× |
| 8, 512 | 1.74 | 1.72 | tie |
| 8, 2048 | 21.97 | 32.17 | SK 1.5× |
| 16, 512 | 2.88 | 3.92 | SK 1.4× |
| 8, 4096 | 94.92 | 164.70 | **SK 1.7×** |

Why: ds4's `_vec` kernel is decode-optimized (NQPSG=1, one query per
threadgroup); SK's `mha_causal` processes Br=4 queries / TG and amortizes K/V
loads. **Don't consolidate.** Each kernel optimal for its operating point.

Bench scripts under `temp/flash_attn_compare/` (preserved for re-runs).

## Public API

C ABI: `sk_flash_attn_ext_vec(Q, K, V, mask, O, B, H, H_kv, S_q, S_kv, dk, dv, has_mask, scale, nsg, nwg)`.
Python: `from kernels.flash_attn.flash_attn import flash_attn_ext_vec, causal_mask`.

Verified bit-exact (max-err 0.0) at:
- d=128: (B=1, H=8, S=512) causal MHA
- d=512: (B=1, H=4, S=64) prefill, and (B=1, H=4, S_q=1, S_kv=64) decode

## (dk, dv) instantiations currently in libsk.metallib

- `kernel_flash_attn_ext_vec_f16_dk128_dv128`
- `kernel_flash_attn_ext_vec_f16_dk512_dv512`

Add another by appending one line at the bottom of `flash_attn.metal`,
e.g. `template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 192, 128, 1>;`
