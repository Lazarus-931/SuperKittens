//
//  launcher.h — IBM Granite-4.x hybrid (interleaved mamba2 + attention) C ABI.
//
//  Layer types come from GGUF metadata at load time
//  (granitehybrid.attention.head_count_kv: per-layer kv-head count, 0 = mamba).
//  Every layer additionally carries a dense SwiGLU FFN.

#ifndef SK_GRANITE_LAUNCHER_H
#define SK_GRANITE_LAUNCHER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t batch;        // serving lanes (1 = single-stream stage 1)
    uint32_t seq_max;      // max tokens per forward (prompt must fit ONE prefill:
                           // chunked mamba prefill would zero-pad each chunk's conv left edge)
    uint32_t cache_max;    // attention KV capacity
    uint32_t n_layers;     // 40
    uint32_t d_model;      // 1536
    // attention layers
    uint32_t n_heads;      // 12
    uint32_t n_kv_heads;   // 4
    uint32_t head_dim;     // 128 (mha_causal D=128 instantiation)
    // dense FFN (every layer)
    uint32_t n_int;        // 4096
    // mamba2 layers
    uint32_t d_inner;      // 3072 (E)
    uint32_t ssm_n_heads;  // 48  (H)
    uint32_t ssm_head_dim; // 64  (P)
    uint32_t ssm_state;    // 128 (N)
    uint32_t ssm_n_groups; // 1   (G)
    uint32_t ssm_conv;     // 4   (K)
    uint32_t vocab_size;   // 100352
    float    eps;          // 1e-5
    // granite multipliers (GGUF: granitehybrid.{embedding,residual,attention,logit}_scale)
    float    embedding_scale;  // 12.0
    float    residual_scale;   // 0.22
    float    attention_scale;  // 0.0078125 (replaces 1/sqrt(head_dim))
    float    logit_scale;      // 6.0 (logits are DIVIDED by this)
} sk_granite_config;

typedef struct sk_granite_handle sk_granite_handle;

sk_granite_handle* sk_granite_create(const sk_granite_config* cfg);

// Loads weights AND the per-layer type list from GGUF metadata. Q8_0 + F32
// GGUFs only (granite official Q8_0).
int sk_granite_load_gguf(sk_granite_handle* h, const char* path);

// One forward (prefill seq>1 or decode seq==1); greedy argmax -> *output_id.
int sk_granite_forward(sk_granite_handle* h,
                       const int* input_ids, uint32_t seq, int* output_id);

// Lockstep batched decode (cfg.batch = N serving lanes): input_ids[batch] (one
// token per lane), output_id[batch] greedy next tokens. All lanes advance from
// the same current_pos; each reads/writes its own KV-cache slice (4 attention
// layers) and conv/ssm state slice (36 mamba2 layers). seq must be 1; lockstep
// prompt ingestion is either token-by-token through this entrypoint or one
// sk_granite_prefill_batched call. Requires cfg.batch == N at create.
int sk_granite_forward_batched(sk_granite_handle* h,
                               const int* input_ids, uint32_t seq,
                               int* output_id);

// Batched-lane prefill: all cfg.batch lanes' EQUAL-LENGTH prompts (lane-major
// ids[batch*seq]) in ONE batch=N forward — weights read once instead of once
// per lane per token. Same stage-1 scope as sk_granite_forward: the prompt
// must fit one forward (seq <= seq_max; chunking would re-zero-pad the mamba
// conv left edge), and the handle must be fresh (current_pos == 0; mamba state
// demands a fresh sequence). output_ids[0..batch) = per-lane greedy first
// tokens; lane states are left ready for sk_granite_forward_batched decode.
int sk_granite_prefill_batched(sk_granite_handle* h,
                               const int* ids, uint32_t seq, int* output_ids);

// fp16 logits row of lane `lane` after a batched forward (vocab_size entries,
// post logit_scale). After single-stream calls only lane 0 is valid.
int sk_granite_get_logits_row(sk_granite_handle* h, uint32_t lane,
                              void* out_fp16);

// Greedy loop in C: prefill prompt then decode n_tokens (stops at eos_id >= 0).
// Returns number of tokens written, or negative error.
int sk_granite_generate_n(sk_granite_handle* h,
                          const int* prompt_ids, uint32_t prompt_seq,
                          int* out_tokens, uint32_t n_tokens, int32_t eos_id);

// fp16 logits of the last projected row (vocab_size entries, post logit_scale).
int sk_granite_get_last_logits(sk_granite_handle* h, void* out_fp16);

uint32_t sk_granite_get_pos(sk_granite_handle* h);

// Zero attention position + all mamba conv/ssm state.
void sk_granite_reset(sk_granite_handle* h);

void sk_granite_destroy(sk_granite_handle* h);

#ifdef __cplusplus
}
#endif

#endif
