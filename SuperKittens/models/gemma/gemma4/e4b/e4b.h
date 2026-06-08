//
//  e4b.h — Gemma 4 E4B variant config preset.
//
//  Source: google/gemma-4-E4B-it config.json (text_config), cross-checked.
//
//  E4B = "Effective 4B". Same gemma4 arch as E2B (PLE, 6-layer SWA period
//  with full_attention at L%6==5) but 42 layers, wider hidden/MLP, GQA 4:1
//  (8 Q / 2 KV heads), 18 KV-shared tail layers, and double-wide MLP OFF
//  (E2B has it ON). The Python adapter reads all of these from config.json
//  via _build; this preset is the C-ABI fallback.
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
    c.n_layers           = 42;
    c.local_period       = 6;         // full_attention at L%6==5 (L5,11,...,41)
    c.d_model            = 2560;
    c.n_int              = 10240;
    c.n_heads            = 8;
    c.n_kv_heads_local   = 2;         // GQA 4:1
    c.n_kv_heads_global  = 2;
    c.head_dim_local     = 256;
    c.head_dim_global    = 512;
    c.window             = 512;       // real sliding_window
    c.prope_p_pairs      = 64;
    c.vocab_size         = 262144;
    c.has_ple            = 1;
    c.eps                = 1e-6f;
    c.final_logit_softcap   = 30.0f;
    c.use_double_wide_mlp   = 0;      // E4B: double-wide MLP OFF (E2B: ON)
    c.num_kv_shared_layers  = 18;
    return c;
}

#ifdef __cplusplus
}
#endif

#endif
