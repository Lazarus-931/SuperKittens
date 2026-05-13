//
//  e2b.h — Gemma 4 E2B variant config preset.
//
//  Source: https://ai.google.dev/gemma/docs/core/model_card_4
//  Numbers cross-checked against the HuggingFace model card and the
//  community write-ups (Maarten Grootendorst, Kaitchup) on the architecture.
//
//  E2B = "Effective 2B". Real param count after PLE accounting is ~2.3B.
//  Designed for on-device inference (mobile, edge); has Per-Layer Embeddings
//  (PLE) which contribute parameters that don't appear in the dense weights
//  but recover model capacity when looked up per-layer.
//

#ifndef SK_GEMMA4_E2B_H
#define SK_GEMMA4_E2B_H

#include "../launcher.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline sk_gemma4_config gemma4_e2b_config(void) {
    sk_gemma4_config c;
    c.batch              = 1;
    c.seq_max            = 256;       // prefill cap: per-dispatch overhead scales with backing
                                      // buffer size on Apple GPUs, so we size scratch for the
                                      // common chat-prefill regime. Raise via override for long prompts.
    c.cache_max          = 8192;      // global KV cache cap (raise if more memory available)
    c.n_layers           = 35;
    c.local_period       = 6;         // 5 local : 1 global
    c.d_model            = 1536;
    c.n_int              = 6144;
    c.n_heads            = 8;         // E2B has 8 attention heads (HF config)
    c.n_kv_heads_local   = 1;         // GQA ratio 8:1
    c.n_kv_heads_global  = 1;
    c.head_dim_local     = 256;
    c.head_dim_global    = 512;
    c.window             = 4096;      // E-models: 4K SWA
    c.prope_p_pairs      = 64;
    c.vocab_size         = 262144;
    c.has_ple            = 1;
    c.eps                = 1e-6f;
    return c;
}

#ifdef __cplusplus
}
#endif

#endif
