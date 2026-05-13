//
//  e4b.h — Gemma 4 E4B variant config preset.
//
//  Source: https://ai.google.dev/gemma/docs/core/model_card_4
//
//  E4B = "Effective 4B". Real total ~4.5B with PLE. Same architectural
//  family as E2B (same context, same window, has PLE) — just bigger
//  hidden + more attention heads.
//

#ifndef SK_GEMMA4_E4B_H
#define SK_GEMMA4_E4B_H

#include "../launcher.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline sk_gemma4_config gemma4_e4b_config(void) {
    sk_gemma4_config c;
    c.batch              = 1;
    c.seq_max            = 256;       // see e2b.h: scratch-buffer size dominates per-dispatch overhead
    c.cache_max          = 8192;
    c.n_layers           = 35;
    c.local_period       = 6;
    c.d_model            = 2048;
    c.n_int              = 8192;
    c.n_heads            = 8;
    c.n_kv_heads_local   = 8;         // local: full per-Q KV
    c.n_kv_heads_global  = 2;         // global: 4:1 GQA
    c.head_dim_local     = 256;
    c.head_dim_global    = 512;
    c.window             = 4096;
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
