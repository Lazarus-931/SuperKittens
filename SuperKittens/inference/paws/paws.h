//
//  paws.h — KV cache block allocator for Apple Silicon inference
//
//  Usage:
//    paws_config cfg = {.block_tokens=16, .num_kv_heads=32, .head_dim=128};
//    paws_ctx* ctx = paws_create(&cfg);
//    paws_seq_init(ctx, seq_id);
//    int32_t blk = paws_seq_append(ctx, seq_id);  // one per 16 tokens
//    const int32_t* table = paws_seq_blocks(ctx, seq_id); // for kernel
//    paws_seq_free(ctx, seq_id);
//    paws_destroy(ctx);
//

#ifndef PAWS_H
#define PAWS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── config (set once, matches model config.json) ──

typedef struct {
    uint32_t block_tokens;      // tokens per block (default 16)
    uint32_t num_kv_heads;      // GQA-aware (0 = no KV cache, e.g. Mamba)
    uint32_t head_dim;          // e.g. 64, 128, 256
    uint32_t precision_bytes;   // 2=fp16, 1=int8, 1=packed int4
    uint32_t max_blocks;        // total pool capacity
} paws_config;

// ── opaque context ──

typedef struct paws_ctx paws_ctx;

// ── lifecycle ──

paws_ctx* paws_create(const paws_config* cfg);
void      paws_destroy(paws_ctx* ctx);

// ── raw block alloc (caller rarely uses directly) ──

int32_t   paws_block_alloc(paws_ctx* ctx);
void      paws_block_free(paws_ctx* ctx, int32_t block_idx);

// ── sequence tracking ──

void      paws_seq_init(paws_ctx* ctx, uint32_t seq_id);
int32_t   paws_seq_append(paws_ctx* ctx, uint32_t seq_id);
void      paws_seq_free(paws_ctx* ctx, uint32_t seq_id);

// ── block table for kernel dispatch ──

const int32_t* paws_seq_blocks(const paws_ctx* ctx, uint32_t seq_id);
uint32_t       paws_seq_len(const paws_ctx* ctx, uint32_t seq_id);

// ── derived values (for kernel dispatch) ──

uint32_t paws_block_tokens(const paws_ctx* ctx);
uint32_t paws_block_elements(const paws_ctx* ctx);    // block_tokens * num_kv_heads * head_dim
uint32_t paws_block_bytes(const paws_ctx* ctx);        // elements * precision_bytes

// ── direct buffer access (zero-copy for kernel) ──

void*   paws_buffer_ptr(paws_ctx* ctx);                 // MTL::Buffer contents()
void*   paws_block_ptr(paws_ctx* ctx, int32_t idx);     // &buffer[idx * block_bytes]

// ── stats ──

uint32_t paws_num_free(const paws_ctx* ctx);
uint32_t paws_num_used(const paws_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif
