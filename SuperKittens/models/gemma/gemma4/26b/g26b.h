//
//  g26b.h — Gemma 4 26B-A4B variant config preset.
//
//  Source: https://ai.google.dev/gemma/docs/core/model_card_4
//          https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture
//          https://huggingface.co/google/gemma-4-26B-A4B
//
//  26B-A4B = "26B total / 4B active per token" Mixture-of-Experts.
//  - 128 experts per MoE layer; top-8 selected per token (one of which is a
//    shared always-on expert per the architecture description).
//  - Routes through models/gemma/gemma4/26b/moe_block.h instead of the dense
//    GeGLU MLP step in gemma4_model.h::dispatch_layer.
//  - Sliding window 1024 (NOT the 4096 that E-models use).
//  - No PLE.
//

#ifndef SK_GEMMA4_26B_H
#define SK_GEMMA4_26B_H

#include "../launcher.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline sk_gemma4_config gemma4_26b_config(void) {
    sk_gemma4_config c;
    c.batch              = 1;
    c.seq_max            = 8192;
    c.cache_max          = 8192;
    c.n_layers           = 60;
    c.local_period       = 6;
    c.d_model            = 4608;
    c.n_int              = 12288;     // per-expert FFN intermediate
    c.n_heads            = 16;
    c.n_kv_heads_local   = 16;
    c.n_kv_heads_global  = 4;
    c.head_dim_local     = 256;
    c.head_dim_global    = 512;
    c.window             = 1024;      // 26B/31B: 1K SWA
    c.prope_p_pairs      = 64;
    c.vocab_size         = 262144;
    c.has_ple            = 0;
    c.eps                = 1e-6f;
    return c;
}

// MoE-specific config (separate from sk_gemma4_config which is dense).
// Used by moe_block.h dispatcher when running 26B-A4B layers.
typedef struct {
    uint32_t n_expert;       // 128
    uint32_t top_k;          // 8 (includes 1 shared expert)
    int      has_shared;     // 1 — Gemma 4 26B-A4B has a shared always-on expert
} sk_gemma4_26b_moe;

static inline sk_gemma4_26b_moe gemma4_26b_moe_config(void) {
    sk_gemma4_26b_moe m;
    m.n_expert   = 128;
    m.top_k      = 8;
    m.has_shared = 1;
    return m;
}

#ifdef __cplusplus
}
#endif

#endif
