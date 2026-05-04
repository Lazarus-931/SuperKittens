//
//  paws.c++ — KV cache block allocator implementation
//

#include "paws.h"
#include <Metal/Metal.hpp>
#include <vector>
#include <cstring>

struct paws_ctx {
    MTL::Device*  device;
    MTL::Heap*    heap;
    MTL::Buffer*  buffer;

    uint32_t      block_tokens;
    uint32_t      num_kv_heads;
    uint32_t      head_dim;
    uint32_t      precision_bytes;
    uint32_t      block_bytes;
    uint32_t      max_blocks;
    int32_t       free_head;

    std::vector<std::vector<int32_t>> seq_blocks;
    mutable std::vector<int32_t>      scratch;
};

paws_ctx* paws_create(const paws_config* cfg) {
    if (!cfg || cfg->num_kv_heads == 0) return nullptr;

    auto* ctx  = new paws_ctx{};
    ctx->block_tokens    = cfg->block_tokens ? cfg->block_tokens : 16;
    ctx->num_kv_heads    = cfg->num_kv_heads;
    ctx->head_dim        = cfg->head_dim;
    ctx->precision_bytes = cfg->precision_bytes ? cfg->precision_bytes : 2;
    ctx->max_blocks      = cfg->max_blocks;
    ctx->free_head       = -1;

    // block_bytes = block_tokens * num_kv_heads * head_dim * precision_bytes
    uint64_t elements = (uint64_t)ctx->block_tokens * ctx->num_kv_heads * ctx->head_dim;
    ctx->block_bytes = (uint32_t)(elements * ctx->precision_bytes);

    uint64_t heap_size = (uint64_t)ctx->block_bytes * ctx->max_blocks;
    ctx->device = MTL::CreateSystemDefaultDevice();

    auto* desc = MTL::HeapDescriptor::alloc()->init();
    desc->setSize(heap_size);
    desc->setStorageMode(MTL::StorageModeShared);
    ctx->heap   = ctx->device->newHeap(desc);
    ctx->buffer = ctx->heap->newBuffer(heap_size, MTL::ResourceStorageModeShared);
    desc->release();

    // Inline free list: each free block's first 4 bytes = next index
    ctx->free_head = 0;
    for (int32_t i = 0; i < (int32_t)ctx->max_blocks - 1; i++) {
        *(int32_t*)((char*)ctx->buffer->contents() + (size_t)i * ctx->block_bytes) = i + 1;
    }
    if (ctx->max_blocks > 0) {
        *(int32_t*)((char*)ctx->buffer->contents() +
                    (size_t)(ctx->max_blocks - 1) * ctx->block_bytes) = -1;
    }
    return ctx;
}

void paws_destroy(paws_ctx* ctx) {
    if (!ctx) return;
    ctx->buffer->release();
    ctx->heap->release();
    delete ctx;
}

// ── block ops ──

int32_t paws_block_alloc(paws_ctx* ctx) {
    if (ctx->free_head < 0) return -1;
    int32_t idx = ctx->free_head;
    ctx->free_head = *(int32_t*)((char*)ctx->buffer->contents() + (size_t)idx * ctx->block_bytes);
    return idx;
}

void paws_block_free(paws_ctx* ctx, int32_t idx) {
    if (idx < 0) return;
    *(int32_t*)((char*)ctx->buffer->contents() + (size_t)idx * ctx->block_bytes) = ctx->free_head;
    ctx->free_head = idx;
}

// ── sequence ops ──

void paws_seq_init(paws_ctx* ctx, uint32_t seq_id) {
    if (seq_id >= ctx->seq_blocks.size()) ctx->seq_blocks.resize(seq_id + 1);
    ctx->seq_blocks[seq_id].clear();
}

int32_t paws_seq_append(paws_ctx* ctx, uint32_t seq_id) {
    int32_t blk = paws_block_alloc(ctx);
    if (blk < 0) return -1;
    if (seq_id >= ctx->seq_blocks.size()) ctx->seq_blocks.resize(seq_id + 1);
    ctx->seq_blocks[seq_id].push_back(blk);
    return blk;
}

void paws_seq_free(paws_ctx* ctx, uint32_t seq_id) {
    if (seq_id >= ctx->seq_blocks.size()) return;
    for (int32_t blk : ctx->seq_blocks[seq_id]) paws_block_free(ctx, blk);
    ctx->seq_blocks[seq_id].clear();
}

// ── queries ──

const int32_t* paws_seq_blocks(const paws_ctx* ctx, uint32_t seq_id) {
    if (seq_id >= ctx->seq_blocks.size()) return nullptr;
    auto& v = ctx->seq_blocks[seq_id];
    ctx->scratch.assign(v.begin(), v.end());
    return ctx->scratch.data();
}

uint32_t paws_seq_len(const paws_ctx* ctx, uint32_t seq_id) {
    if (seq_id >= ctx->seq_blocks.size()) return 0;
    return (uint32_t)ctx->seq_blocks[seq_id].size();
}

uint32_t paws_block_tokens(const paws_ctx* ctx) { return ctx->block_tokens; }
uint32_t paws_block_elements(const paws_ctx* ctx) {
    return ctx->block_tokens * ctx->num_kv_heads * ctx->head_dim;
}
uint32_t paws_block_bytes(const paws_ctx* ctx) { return ctx->block_bytes; }

void* paws_buffer_ptr(paws_ctx* ctx) { return ctx->buffer->contents(); }
void* paws_block_ptr(paws_ctx* ctx, int32_t idx) {
    return (char*)ctx->buffer->contents() + (size_t)idx * ctx->block_bytes;
}

uint32_t paws_num_free(const paws_ctx* ctx) {
    uint32_t n = 0;
    for (int32_t cur = ctx->free_head; cur >= 0;
         cur = *(int32_t*)((char*)ctx->buffer->contents() + (size_t)cur * ctx->block_bytes)) n++;
    return n;
}

uint32_t paws_num_used(const paws_ctx* ctx) {
    uint32_t n = 0;
    for (auto& s : ctx->seq_blocks) n += (uint32_t)s.size();
    return n;
}
