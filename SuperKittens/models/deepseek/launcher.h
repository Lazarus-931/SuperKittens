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
    // 1 = INT2_DS4 (IQ2_XXS up/gate + Q2_K down — DS4 V4 Flash production),
    // 2 = Q4_K (V2-Lite Q4_K_M routed experts, per-expert mul_mv_id_q4_K).
    uint32_t moe_quant;

    int32_t  rope_n_ctx_orig;
    float    rope_freq_base;
    float    rope_freq_scale;
    float    rope_ext_factor;
    float    rope_attn_factor;
    float    rope_beta_fast;
    float    rope_beta_slow;

    float    eps;

    // V3 additions. Order is load-bearing — Python ctypes must mirror exactly.
    uint32_t has_q_lora;             // V2-Lite: 0
    uint32_t router_has_bias;        // V2-Lite: 0
    uint32_t rope_interleave;        // V3: 1 (interleaved/GPT-J-pair RoPE)
    uint32_t norm_topk_prob;         // V2-Lite: 0
    uint32_t n_group;                // V2-Lite: 0 (disables grouping)
    uint32_t topk_group;
    float    routed_scaling_factor;  // V2-Lite: 1.0
    float    mscale_all_dim;         // YaRN mscale_all_dim
    float    rope_scaling_factor;    // YaRN factor (1.0 = disabled)
    uint32_t first_k_dense_replace;  // V3: 3, V2-Lite: 1
    uint32_t dense_n_int;            // leading-dense-layer MLP intermediate (V2-Lite 10944)
} sk_deepseek_config;

// Host weight pointers (NULL allowed; create allocates zero-init buffers).
typedef struct {
    const void* w_embed;             // (vocab, d_model)
    const void* w_lm_head;           // (vocab, d_model) — optional; null = tied (alias w_embed)
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
    const void* router_bias;         // (n_layers, n_expert) fp32 — optional, V3 only
    const void* w_gate;              // (n_layers, n_expert, n_int, d_model)
    const void* w_up;                // (n_layers, n_expert, n_int, d_model)
    const void* w_down;              // (n_layers, n_expert, d_model, n_int)
} sk_deepseek_weights;

sk_deepseek_handle* sk_deepseek_create(const sk_deepseek_config* cfg);
int  sk_deepseek_load_weights(sk_deepseek_handle* h, const sk_deepseek_weights* w);
int  sk_deepseek_forward(sk_deepseek_handle* h,
                         const int* input_ids, uint32_t seq, int* output_id);
// Batched lockstep decode: `batch` requests step together at current_pos, each
// with its own KV region. input_ids = batch int32 (one token/lane), output_id =
// batch int32 (one greedy next-token/lane). batch==1 == single-stream forward.
int  sk_deepseek_forward_batched(sk_deepseek_handle* h,
                                 const int* input_ids, int* output_id);
// Batched chunked prefill: ids = batch*seq int32 (request-major, equal-length
// lockstep lanes), chunks of <= chunk_size (0 = seq_max) per dispatch; logits/
// argmax only on the final chunk. out_next = batch int32 greedy next tokens.
int  sk_deepseek_prefill_batched(sk_deepseek_handle* h,
                                 const int* ids, uint32_t seq,
                                 uint32_t chunk_size, int* out_next);
void sk_deepseek_reset(sk_deepseek_handle* h);
void sk_deepseek_destroy(sk_deepseek_handle* h);

// Optional: print the SK_DS_PROFILE per-category GPU-time breakdown for the
// given token count and reset the accumulators. No-op when profiling is off.
void sk_deepseek_profile_report(uint64_t tokens);

#ifdef __cplusplus
}
#endif
#endif
