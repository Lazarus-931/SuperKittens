# Phase 2 — per-token setBytes inventory (qwen3 + gemma4)

Pre-flight survey: every `setBytes` call inside the per-token decode loop
that targets a scalar which CHANGES per token (vs. model-static shapes
that could be hoisted to once-per-handle).

## Per-token scalars (must move to `token_args` MTLBuffer)

| Model  | File / line                    | Kernel (host_name)      | Slot | Scalar           |
|--------|--------------------------------|-------------------------|------|------------------|
| qwen3  | qwen_model.h:392               | kv_cache_write          | 8    | write_pos        |
| qwen3  | qwen_model.h:413               | mha_causal (P.attn)     | 7    | kv_len           |
| qwen3  | qwen_model.h: rope (cs_off)    | rope_qk                 | n/a  | write_pos*       |
| qwen3  | launcher.c++ pre-decode        | embedding_lookup        | n/a  | input_ids[0]**   |
| gemma4 | gemma4_model.h:308             | rms_rope (fused)        | 12   | write_pos        |
| gemma4 | gemma4_model.h:375/387/399/410 | rope_q / rope_k         | 8/7  | write_pos        |
| gemma4 | gemma4_model.h: attn / kv_w    | kv_cache_write / attn   | (mirrors qwen3) | write_pos, kv_len |

\* RoPE in qwen3 is fed via `cos_tbl` / `sin_tbl` byte offset
(`cs_off = write_pos * (hd/2) * 2`) rather than a scalar argument; this is
moved by patching the *MTL::Buffer offset* in the recorded ICB slot, not by
patching `token_args`. Listed for completeness — slot-offset patch is a
recorder-level concern.

\*\* token-id reaches `embedding_lookup` via the existing `B.input_ids`
buffer, which the host already memcpy-patches per token. No setBytes
involved; included so the inventory is complete.

## Model-static scalars (NOT migrated — set once at handle create)

`n_heads`, `n_kv_heads`, `head_dim`, `d_model`, `n_int`, `vocab_size`,
`eps`, `rope_*`, `cache_size`, `n_layers`, GEMM shapes (M/N/K/ld*),
`transA`/`transB`/`has_bias`. These are set every token today via
`setBytes` but are constant for a given handle; Phase 2 hoists them into
once-recorded ICB slots via small per-kernel "shape args" MTLBuffers as a
follow-up (Phase 2.5), since they are *not* the ICB blocker.

## TokenArgs schema (32 bytes, scalar-only)

```cpp
namespace sk {
struct TokenArgs {                 // offset (B)
    uint32_t current_pos;          //  0
    uint32_t kv_idx_base;          //  4
    int32_t  token_id;             //  8
    uint32_t layer_idx;            // 12
    uint32_t reserved[4];          // 16..32
};                                 // sizeof == 32
} // namespace sk
```

Patch protocol (host, per token, no encoder):
```cpp
sk::TokenArgs a = {pos, kv_base, tok, /*layer=*/0, {0,0,0,0}};
std::memcpy(B.token_args->contents(), &a, sizeof(a));
```

## Kernel migration plan (deferred to follow-up; SEE BELOW)

Phase 2 STOP condition fired: every kernel listed above
(`kv_cache_write`, `attn`/`mha_causal`, `rope_qk`, `rms_rope`,
`embedding_lookup`) is a **shared kernel** consumed by deepseek-v2-lite
and the mamba launchers as well. Spec rule:

> "If a kernel signature change breaks something downstream (other
>  models that share that kernel), STOP and report."

We do not blanket-break shared kernel signatures in this PR. The
unblocking path (deferred to a follow-up PR, REQUIRES BENCH on mini):

1. Fork per-model ICB variants:
   `kv_cache_write_icb` / `mha_causal_icb` / `rope_qk_icb` that read the
   per-token scalars from `constant TokenArgs& [[buffer(N)]]` instead of
   inline `setBytes` slots, leaving the original kernels untouched for
   deepseek + mamba.
2. Bind those variants only on qwen3 + gemma4 ICB-tail launch paths.
3. Bit-exact diff vs. the legacy non-ICB path under bench.

This PR delivers the **plumbing surface** (`TokenArgs` struct, host
patch API, `token_args` buffer wired into both launchers, recorder
already exists) so the follow-up is mechanical.
