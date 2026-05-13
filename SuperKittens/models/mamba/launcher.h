#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// C ABI for SK Mamba 1 (state-spaces/mamba-*-hf). Scaffold only — see
// STATUS.md for which entry points are implemented vs pending.

typedef struct sk_mamba_ctx sk_mamba_ctx;

sk_mamba_ctx* sk_mamba_create(const char* weights_dir);
void          sk_mamba_destroy(sk_mamba_ctx* ctx);

// Prefill / decode entry. tokens: int32 ids, n_tokens. logits_out: fp32 [vocab].
int sk_mamba_forward(sk_mamba_ctx* ctx,
                     const int32_t* tokens,
                     uint32_t n_tokens,
                     float* logits_out);

// Reset SSM + conv state caches between independent prompts.
void sk_mamba_reset_cache(sk_mamba_ctx* ctx);

// Validation hook: write per-layer activations (matching keys produced by
// dump_hf_mamba.py) into an npz at `path`. No-op until kernels land.
int sk_mamba_dump_layer(sk_mamba_ctx* ctx, const char* path);

#ifdef __cplusplus
}
#endif
