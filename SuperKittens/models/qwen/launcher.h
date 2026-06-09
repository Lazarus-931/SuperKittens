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
    uint32_t use_qk_norm;          // 1 = Qwen3 per-head Q/K RMSNorm; 0 = Llama-arch (Nemotron-Nano-8B) no qk-norm
    uint32_t rope_interleaved;     // 0 = split-half/NeoX RoPE (GGML type 2, Qwen3); 1 = interleaved/NORM (GGML type 0, Llama GGUF)
    uint32_t attn_qkv_bias;        // 1 = Qwen2/Qwen2.5 additive q/k/v projection bias; 0 = no bias (Qwen3/Llama/Mistral)
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
    const void* w_qkv_bias; // optional; (n_layers, qkv_N) packed [Q|K|V] bias. null when attn_qkv_bias=0
} sk_qwen_weights;

sk_qwen_handle* sk_qwen_create(const sk_qwen_config* cfg);
int  sk_qwen_load_weights(sk_qwen_handle* h, const sk_qwen_weights* w);
int  sk_qwen_forward(sk_qwen_handle* h,
                     const int* input_ids, uint32_t seq, int* output_id);
// Batched lockstep decode: input_ids is batch*seq int32 (request-major), output_id
// receives `batch` greedy next tokens (one per request). All requests advance from
// the same absolute position; each reads/writes its own KV-cache slice. Requires
// the handle to have been created with cfg.batch == N.
int  sk_qwen_forward_batched(sk_qwen_handle* h,
                             const int* input_ids, uint32_t seq, int* output_id);
// Chunked prefill: process prompt_ids in fixed-size chunks (<= chunk_size, and
// <= seq_max) through the per-step model forward, carrying KV cache + position
// across chunks. Lets a prompt longer than seq_max prefill with scratch bounded
// to one chunk (memory grows with chunk_size, not prompt_seq). On return,
// output_id holds the greedy next token and get_last_logits the final-position
// logits — identical to a single seq=prompt_seq forward. chunk_size 0 → seq_max.
int  sk_qwen_prefill_chunked(sk_qwen_handle* h,
                             const int* prompt_ids, uint32_t prompt_seq,
                             uint32_t chunk_size, int* output_id);
// WHY: per-token decode loop kept in C; avoids ctypes round-trip + np alloc
// per step. Greedy only. Returns count of tokens written into out_tokens
// (<= n_tokens; stops early on eos_id when eos_id >= 0).
int  sk_qwen_generate_n(sk_qwen_handle* h,
                        const int* prompt_ids, uint32_t prompt_seq,
                        int* out_tokens, uint32_t n_tokens,
                        int32_t eos_id);
void sk_qwen_reset(sk_qwen_handle* h);
void sk_qwen_destroy(sk_qwen_handle* h);

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
