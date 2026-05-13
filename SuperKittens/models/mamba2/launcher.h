//  launcher.h — single C ABI for Mamba 2 (mamba2-130m-hf) inference.
//
//  Mirrors qwen/launcher.h structure. Mamba 2 has no KV cache; instead per-layer
//  state = (conv_state[H,P,d_conv-1] equivalent: C_in flat conv state of
//  length d_conv-1, where C_in = E + 2*G*N) + (ssm_state[H,P,N]).
//
//  Reference: transformers/models/mamba2/modeling_mamba2.py (HF).
//  Config sourced from AntonV/mamba2-130m-hf/config.json:
//    hidden_size=768, num_hidden_layers=24, num_heads=24, head_dim=64,
//    state_size=128, expand=2, intermediate_size=1536, n_groups=1,
//    chunk_size=256, conv_kernel=4, time_step_rank=256,
//    use_bias=false, use_conv_bias=true, rms_norm=true,
//    vocab_size=50288, tie_word_embeddings=true.

#ifndef SK_MAMBA2_LAUNCHER_H
#define SK_MAMBA2_LAUNCHER_H

#include <cstdint>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_mamba2_handle sk_mamba2_handle;

typedef struct {
    uint32_t batch;
    uint32_t seq_max;        // max prefill length
    uint32_t n_layers;       // 24
    uint32_t d_model;        // 768
    uint32_t intermediate;   // 1536 (E = expand*d_model)
    uint32_t n_heads;        // 24 (H)
    uint32_t head_dim;       // 64 (P)
    uint32_t state_size;     // 128 (N)
    uint32_t n_groups;       // 1 (G)
    uint32_t conv_kernel;    // 4
    uint32_t chunk_size;     // 256
    uint32_t vocab_size;     // 50288
    float    rms_eps;        // 1e-5 (HF default)
    float    time_step_min;  // softplus dt clamp lower (HF default 0.001)
    float    time_step_max;  // softplus dt clamp upper (HF default 0.1)
    uint32_t tie_word_embeddings;  // 1
} sk_mamba2_config;

typedef struct {
    // shared
    const void* w_embed;          // (vocab, d_model)
    const void* w_final_norm;     // (d_model,)
    // per-layer (n_layers, ...)
    const void* w_pre_norm;       // (n_layers, d_model)            — backbone.layers.L.norm.weight
    const void* w_in_proj;        // (n_layers, d_model, IN_PROJ_OUT)
                                  //   IN_PROJ_OUT = 2*E + 2*G*N + H
                                  //   layout: [z(E) | x(E) | B(G*N) | C(G*N) | dt(H)]
    const void* w_conv;           // (n_layers, conv_kernel, C_in) C_in = E + 2*G*N
    const void* w_conv_b;         // (n_layers, C_in)
    const void* w_dt_bias;        // (n_layers, H)
    const void* w_A_log;          // (n_layers, H)
    const void* w_D;              // (n_layers, H)
    const void* w_norm;           // (n_layers, E)   mixer.norm (gated RMSNorm γ)
    const void* w_out_proj;       // (n_layers, E, d_model)
} sk_mamba2_weights;

// Per-layer running state for decode.
typedef struct {
    void* conv_state;             // (batch, conv_kernel-1, C_in)  fp16
    void* ssm_state;              // (batch, H, P, N)              fp16
} sk_mamba2_layer_state;

sk_mamba2_handle* sk_mamba2_create(const sk_mamba2_config* cfg);
int  sk_mamba2_forward(sk_mamba2_handle* h,
                       const int* input_ids, uint32_t seq, int* output_id);
// Dump intermediate tensor by name for HF parity testing. Names mirror
// hf_ref.npz keys: "embed", "L{i}.{mixer_in,xBC_preconv,dt_pre,mixer_out,hidden}", "logits".
int  sk_mamba2_dump_layer(sk_mamba2_handle* h, const char* tag,
                          void* out, size_t out_bytes);
void sk_mamba2_reset(sk_mamba2_handle* h);
void sk_mamba2_destroy(sk_mamba2_handle* h);

#ifdef __cplusplus
}
#endif
#endif
