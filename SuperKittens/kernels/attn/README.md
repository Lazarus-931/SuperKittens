# Kernels

Fused multi-head attention for Apple Silicon. Row-per-SIMD online softmax with cooperative K/V tile loading.

## Kernels

| File | Host name | Type | d=128 fast path |
|---|---|---|---|
| `attn_causal.metal` | `mha_causal` | Causal MHA | half4 vectorized |
| `attn_noncausal.metal` | `mha_noncausal` | Non-causal MHA | half4 vectorized |

Both kernels share the same 7-buffer API:

```
Q, K, V, O, seq, head_dim, num_heads
```

Grid: `(num_heads, ceil_div(seq, 4), 1)`, 128 threads per threadgroup.

For GQA/MQA, add a `num_kv_heads` buffer and compute `kv_head = q_head * num_kv_heads / num_q_heads` — no separate kernel needed.

## Supporting files

| File | Purpose |
|---|---|
| `ops.h` | Tile-based softmax ops: causal mask, scale+max, exp+sum, online state update. Used by both kernels. |
| `params.h` | Host-side `MHA_Params` struct mapping to the 7-buffer API. |
| `types.h` | Tile config structs for future templated kernels with `static_assert` memory validation. |
| `baseline/` | MLX and PyTorch reference implementations and benchmarks. |

## Architecture

Each kernel processes 4 rows per threadgroup (one per SIMD group of 32 threads).

**d=128 fast path**: Q row loaded as `half4`, K/V tiles (32 rows) loaded cooperatively into threadgroup memory. Dot products computed via `simd_sum(dot(float4, float4))` — fully vectorized. Online softmax per row.

**Generic path** (arbitrary d): Scalar element-by-element dot products with `simd_sum` reduction. Fewer barriers, simpler code. Used for non-standard head dimensions.

## Organization principles

- **MHA is the foundation** — both causal and non-causal.
- **GQA/MQA is a dispatch variant** — not a separate kernel. Add `num_kv_heads` to the MHA kernel.
- **MLA is a separate kernel** — fundamentally different math (latent KV compression). Lower priority.
- Follows ThunderKittens' pattern: kernels grouped by operation type, shared ops in `ops.h`, configs in `types.h`.
