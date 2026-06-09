//
//  launcher.h — single C ABI for Gemma 4 inference.
//
//  create / load_weights / forward / destroy. Variant (E2B / E4B / 26B / 31B)
//  selected by the config struct passed to create.
//

#ifndef SK_GEMMA4_LAUNCHER_H
#define SK_GEMMA4_LAUNCHER_H

#include <cstdint>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_gemma4_handle sk_gemma4_handle;

typedef struct {
    uint32_t batch;
    uint32_t seq_max;
    uint32_t cache_max;
    uint32_t n_layers;
    uint32_t local_period;
    uint32_t d_model;
    uint32_t n_int;
    uint32_t n_heads;
    uint32_t n_kv_heads_local;
    uint32_t n_kv_heads_global;
    uint32_t head_dim_local;
    uint32_t head_dim_global;
    uint32_t window;
    uint32_t prope_p_pairs;
    uint32_t vocab_size;
    uint32_t ple_dim;
    int      has_ple;
    float    eps;
    float    final_logit_softcap;
    uint32_t use_double_wide_mlp;
    uint32_t num_kv_shared_layers;
    // gemma4_unified deltas (0 = E-variant gemma4 behaviour):
    //   full_rope_global   — rotate the FULL global head_dim instead of partial 0.25.
    //   apply_layer_scalar — multiply each layer's residual by a per-layer learned
    //                        scalar (GGUF blk.N.layer_output_scale). E-variant only
    //                        applies layer_scalar inside the PLE inject; unified has
    //                        no PLE but still carries this per-layer output scale.
    uint32_t full_rope_global;
    uint32_t apply_layer_scalar;
} sk_gemma4_config;

typedef struct {
    const void* w_embed;
    const void* w_ple_table;
    const void* w_per_layer_input_gate;
    const void* w_per_layer_projection;
    const void* w_layer_scalar;
    const void* w_post_per_layer_input_norm;
    const void* w_per_layer_model_projection;     // (n_layers*ple_dim, d_model)
    const void* w_per_layer_projection_norm;      // (ple_dim,)
    const void* w_pre_attn_norm;
    const void* w_post_attn_norm;
    const void* w_pre_feedforward_layernorm;
    const void* w_post_feedforward_layernorm;
    const void* w_final_norm;
    const void* w_qkv;
    const void* w_out;
    const void* gamma_q;
    const void* gamma_k;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
    const void* cos_local;
    const void* sin_local;
    const void* cos_global;
    const void* sin_global;
} sk_gemma4_weights;

sk_gemma4_handle* sk_gemma4_create(const sk_gemma4_config* cfg);
int sk_gemma4_load_weights(sk_gemma4_handle* h, const sk_gemma4_weights* w);

// Load weights from a GGUF (blk.N.* names, K-quant blocks dequantized to bf16
// on host). Used by the gemma4_unified path: its fit-16GB artifact is a Q4_K_M
// GGUF and the HF bf16 safetensors (23.9 GB) do not fit. Returns 0 on success.
int sk_gemma4_load_gguf(sk_gemma4_handle* h, const char* path);

// Run a forward pass. Maintains an internal current_pos cursor:
//   - Call sk_gemma4_reset before a new sequence (sets current_pos = 0).
//   - Each forward advances current_pos by `seq`.
// Prefill: call once with seq>1. Decode: call repeatedly with seq=1.
int sk_gemma4_forward(sk_gemma4_handle* h,
                      const int* input_ids, uint32_t seq,
                      int* output_id);

// Batched lockstep decode (N concurrent requests share current_pos, each its own
// KV slice). input_ids is batch*seq int32 (request-major), output_id receives
// `batch` greedy next tokens. Requires the handle's cfg.batch == N and seq==1.
// At batch==1 / seq>1 this is undefined; use sk_gemma4_forward there.
int sk_gemma4_forward_batched(sk_gemma4_handle* h,
                              const int* input_ids, uint32_t seq,
                              int* output_id);

// Reset the per-sequence KV-cache cursor (does NOT zero the cache buffers;
// stale data is shadowed once new K/V is written over it).
void sk_gemma4_reset(sk_gemma4_handle* h);

int sk_gemma4_get_last_logits(sk_gemma4_handle* h, void* out_fp16);

// Dump helpers: after a forward(), pull the named tensor's LAST-position
// contents (fp16) from the internal stash. Names supported:
//   "embed"                              — post-embedding (size d_model)
//   "L{L}.x_norm" / "L{L}.attn"
//   "L{L}.mlp"    / "L{L}.out"           — per layer (size d_model)
//   "final_norm"                         — (size d_model)
//   "logits"                             — (size vocab_size, pre-softcap)
//   "L0.q_normed" / "L0.k_normed"        — pre-RoPE per-head fp16 (n_heads*hd_local, n_kv*hd_local)
//   "L0.q_rope"   / "L0.k_rope"          — post-RoPE
//   "L0.attn_pre"                        — pre-o_proj attention output (n_heads*hd_local)
// Returns 0 on success, <0 if name unknown or no stash populated.
// Caller-allocated buffer must be large enough for the named slot.
int sk_gemma4_dump_layer(sk_gemma4_handle* h, const char* name, void* out_fp16);

// Enable/disable per-layer dump stashing. Off by default. When on, each
// forward() populates the internal stash. Adds latency proportional to
// d_model*(4*n_layers + 2) + vocab_size copies.
void sk_gemma4_set_dump_enabled(sk_gemma4_handle* h, int enabled);

// Debug: copy raw bytes from a named weight buffer (host-readable shared mem).
// Names: "w_qkv","w_out","w_pre_attn_norm","w_post_attn_norm",
//        "w_pre_feedforward_layernorm","w_post_feedforward_layernorm",
//        "gamma_q","gamma_k","w_gate","w_up","w_down","w_per_layer_input_gate",
//        "w_per_layer_projection","w_post_per_layer_input_norm","w_layer_scalar".
int sk_gemma4_debug_weight(sk_gemma4_handle* h, const char* name,
                           size_t byte_off, size_t nbytes, void* out);

void sk_gemma4_destroy(sk_gemma4_handle* h);

#ifdef __cplusplus
}
#endif

#endif
