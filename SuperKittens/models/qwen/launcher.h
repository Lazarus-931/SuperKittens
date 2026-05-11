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
} sk_qwen_weights;

sk_qwen_handle* sk_qwen_create(const sk_qwen_config* cfg);
int  sk_qwen_load_weights(sk_qwen_handle* h, const sk_qwen_weights* w);
int  sk_qwen_forward(sk_qwen_handle* h,
                     const int* input_ids, uint32_t seq, int* output_id);
void sk_qwen_reset(sk_qwen_handle* h);
void sk_qwen_destroy(sk_qwen_handle* h);

#ifdef __cplusplus
}
#endif
#endif
