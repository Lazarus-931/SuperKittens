//  launcher.h — single C ABI for Qwen3-32B (dense) inference.

#ifndef SK_QWEN_LAUNCHER_H
#define SK_QWEN_LAUNCHER_H

#include <cstdint>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_qwen_handle sk_qwen_handle;

typedef struct {
    uint32_t batch;
    uint32_t seq_max;
    uint32_t cache_max;
    uint32_t n_layers;
    uint32_t d_model;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t n_int;
    uint32_t vocab_size;

    int32_t  rope_n_ctx_orig;
    float    rope_freq_base;
    float    rope_freq_scale;
    float    rope_ext_factor;
    float    rope_attn_factor;
    float    rope_beta_fast;
    float    rope_beta_slow;

    float    eps;
    uint32_t tie_word_embeddings;  // 1 = tie LM head to embedding (Qwen3-0.6B), 0 = separate lm_head (Qwen3-8B)
} sk_qwen_config;

typedef struct {
    const void* w_embed;
    const void* w_pre_attn_norm;
    const void* w_qkv;
    const void* w_q_norm;          // (n_layers, head_dim) — Qwen3 per-head Q-norm γ
    const void* w_k_norm;          // (n_layers, head_dim) — Qwen3 per-head K-norm γ
    const void* w_o;
    const void* w_pre_mlp_norm;
    const void* w_final_norm;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
    const void* w_lm_head;  // optional (untied); may be null when tie_word_embeddings=1
} sk_qwen_weights;

sk_qwen_handle* sk_qwen_create(const sk_qwen_config* cfg);
int  sk_qwen_load_weights(sk_qwen_handle* h, const sk_qwen_weights* w);
int  sk_qwen_forward(sk_qwen_handle* h,
                     const int* input_ids, uint32_t seq, int* output_id);
// WHY: per-token decode loop kept in C; avoids ctypes round-trip + np alloc
// per step. Greedy only. Returns count of tokens written into out_tokens
// (<= n_tokens; stops early on eos_id when eos_id >= 0).
int  sk_qwen_generate_n(sk_qwen_handle* h,
                        const int* prompt_ids, uint32_t prompt_seq,
                        int* out_tokens, uint32_t n_tokens,
                        int32_t eos_id);
void sk_qwen_reset(sk_qwen_handle* h);
void sk_qwen_destroy(sk_qwen_handle* h);

// Copy the first n_rows RELATIVE rows of the logits buffer (each vocab_size fp16)
// from the most recent forward. Returns -2 if n_rows > seq of that forward.
// WHY: spec-decode verify reads per-position logits for all K verify tokens; the
// LM head writes one full V-row per position at relative rows 0..seq-1.
int  sk_qwen_get_logits_rows(sk_qwen_handle* h, void* out_fp16, uint32_t n_rows);

// KV-cache cursor get/set. WHY: spec-decode rewinds the cursor to discard the KV
// entries of rejected verify tokens. set returns -2 if pos > cache_max.
int  sk_qwen_get_pos(sk_qwen_handle* h);
int  sk_qwen_set_pos(sk_qwen_handle* h, uint32_t pos);

// Debug: limit forward to first N layers (0 = all). Affects subsequent forward calls.
// When < cfg.n_layers, the post-loop final_norm + LM head still run on the
// partial residual stream, so get_last_logits returns logits from the prefix.
int  sk_qwen_set_layers_run(sk_qwen_handle* h, uint32_t n_layers_run);

// Debug: copy the residual stream after layer L into out_fp16
// (shape: (seq, d_model) packed fp16). Returns -1 if L not captured.
// To capture layer L, call sk_qwen_set_capture_layer(h, L) before forward.
int  sk_qwen_set_capture_layer(sk_qwen_handle* h, int32_t layer_idx);
int  sk_qwen_get_capture(sk_qwen_handle* h, void* out_fp16);

#ifdef __cplusplus
}
#endif
#endif
