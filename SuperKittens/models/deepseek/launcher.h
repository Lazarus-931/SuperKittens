//  launcher.h — single C ABI for DeepSeek V4 Flash inference.

#ifndef SK_DEEPSEEK_LAUNCHER_H
#define SK_DEEPSEEK_LAUNCHER_H

#include <cstdint>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_deepseek_handle sk_deepseek_handle;

typedef struct {
    uint32_t batch;
    uint32_t seq_max;
    uint32_t cache_max;

    uint32_t n_layers;
    uint32_t d_model;
    uint32_t n_heads;
    uint32_t qk_nope_dim;       // K's non-rotated half
    uint32_t qk_rope_dim;       // K's RoPE-rotated half (also Q's)
    uint32_t v_head_dim;        // V's head dim
    uint32_t q_lora_rank;
    uint32_t kv_lora_rank;      // compressed-KV dim
    uint32_t n_int;             // per-routed-expert FFN intermediate
    uint32_t shared_n_int;      // shared expert FFN intermediate
    uint32_t n_expert;
    uint32_t top_k;
    uint32_t vocab_size;

    // MoE weight quantization for routed experts. 0 = fp16 (default),
    // 1 = INT2_DS4 (IQ2_XXS up/gate + Q2_K down — DS4 V4 Flash production).
    uint32_t moe_quant;

    int32_t  rope_n_ctx_orig;
    float    rope_freq_base;
    float    rope_freq_scale;
    float    rope_ext_factor;
    float    rope_attn_factor;
    float    rope_beta_fast;
    float    rope_beta_slow;

    float    eps;
} sk_deepseek_config;

// Host weight pointers (NULL allowed; create allocates zero-init buffers).
typedef struct {
    const void* w_embed;             // (vocab, d_model)
    const void* w_pre_attn_norm;     // (n_layers, d_model)
    const void* w_q_a;               // (n_layers, d_model, q_lora_rank)
    const void* w_q_a_norm;          // (n_layers, q_lora_rank)
    const void* w_q_b;               // (n_layers, q_lora_rank, n_heads * (qk_nope_dim + qk_rope_dim))
    const void* w_kv_a;              // (n_layers, d_model, kv_lora_rank + qk_rope_dim)
    const void* w_kv_a_norm;         // (n_layers, kv_lora_rank)
    const void* w_kv_b;              // (n_layers, kv_lora_rank, n_heads * (qk_nope_dim + v_head_dim))
    const void* w_o;                 // (n_layers, n_heads * v_head_dim, d_model)
    const void* w_pre_mlp_norm;      // (n_layers, d_model)
    const void* w_final_norm;        // (d_model,)

    const void* w_shared_gate;       // (n_layers, d_model, shared_n_int)
    const void* w_shared_up;         // (n_layers, d_model, shared_n_int)
    const void* w_shared_down;       // (n_layers, shared_n_int, d_model)

    const void* w_router;            // (n_layers, d_model, n_expert)
    const void* w_gate;              // (n_layers, n_expert, n_int, d_model)
    const void* w_up;                // (n_layers, n_expert, n_int, d_model)
    const void* w_down;              // (n_layers, n_expert, d_model, n_int)
} sk_deepseek_weights;

sk_deepseek_handle* sk_deepseek_create(const sk_deepseek_config* cfg);
int  sk_deepseek_load_weights(sk_deepseek_handle* h, const sk_deepseek_weights* w);
int  sk_deepseek_forward(sk_deepseek_handle* h,
                         const int* input_ids, uint32_t seq, int* output_id);
void sk_deepseek_reset(sk_deepseek_handle* h);
void sk_deepseek_destroy(sk_deepseek_handle* h);

#ifdef __cplusplus
}
#endif
#endif
