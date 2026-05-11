//
//  g31b.h — Gemma 4 31B (dense) variant config preset.
//
//  Source: https://huggingface.co/google/gemma-4-31B
//          https://ai.google.dev/gemma/docs/core/model_card_4
//          https://apxml.com/models/gemma-4-31b
//
//  31B = 30.7B params, dense, 60 decoder layers, hidden 21504, 32 Q heads,
//  16 KV heads, vocab 262144. 256K context. Sliding window 1024 (matches
//  the larger-variant pattern; E-models use 4K instead).
//
//  Largest dense Gemma 4. No PLE.
//

#ifndef SK_GEMMA4_31B_H
#define SK_GEMMA4_31B_H

#include "../launcher.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline sk_gemma4_config gemma4_31b_config(void) {
    sk_gemma4_config c;
    c.batch              = 1;
    c.seq_max            = 8192;
    c.cache_max          = 8192;
    c.n_layers           = 60;
    c.local_period       = 6;
    c.d_model            = 21504;
    c.n_int              = 86016;
    c.n_heads            = 32;
    c.n_kv_heads_local   = 32;        // 31B local: full
    c.n_kv_heads_global  = 16;        // 31B global: 2:1 GQA
    c.head_dim_local     = 256;
    c.head_dim_global    = 512;
    c.window             = 1024;
    c.prope_p_pairs      = 64;
    c.vocab_size         = 262144;
    c.has_ple            = 0;
    c.eps                = 1e-6f;
    return c;
}

#ifdef __cplusplus
}
#endif

#endif
