# paws — scalability analysis & design

## What models need from KV cache

| Model | Heads (H) | Head dim (d) | KV per token | Unique |
|-------|-----------|-------------|--------------|--------|
| Llama 7B | 32 | 128 | 32KB | standard |
| Llama 70B | 64 | 128 | 64KB | standard |
| Mistral 7B | 32 | 128 | 32KB | sliding window (4K) |
| Gemma 7B | 16 | 256 | 32KB | standard |
| Mamba2 7B | — | — | 0 | no KV cache at all |
| Phi-4 | 40 | 128 | 40KB | GQA (8 KV heads) |
| Command-R | 48 | 128 | 24KB | GQA (4 KV heads) |
| Cohere | 32 | 128 | 16KB | GQA (8 KV heads) |

## What paws needs to handle

### 1. Variable block layout
Not all models have the same KV shape. Block = `(block_tokens, num_kv_heads, head_dim)`.
Llama 7B: `(16, 32, 128) * 2 (K+V) * 2 bytes = 256KB/block`
Mistral: same but max 256 blocks due to sliding window

### 2. GQA (Grouped Query Attention)
Q heads ≠ KV heads. KV stored at KV-head granularity, Q replicated.
paws doesn't care — it's just bytes. The paged attention kernel handles GQA.

### 3. Prefill vs decode
- Prefill: process all prompt tokens at once, allocate one block per 16 tokens
- Decode: one token at a time, append to last block or allocate new

### 4. Block size tradeoff
- Small blocks (8 tokens): less fragmentation, more block table overhead
- Large blocks (32 tokens): less overhead, more waste on short sequences
- Default 16 is right for most use cases

### 5. Precision
KV can be fp16, int8, int4. paws just stores bytes — precision is the caller's concern.
But block_size_bytes depends on precision, so it must be configurable.

## Proposed API redesign

Rather than a generic paws with hardcoded assumptions, split into three layers:

```
┌─────────────────────────────────────────┐
│  paws::Config (declarative)             │
│  block_size, max_blocks, precision       │
│  num_kv_heads, head_dim, block_tokens    │
└─────────────────────────────────────────┘
              │
┌─────────────────────────────────────────┐
│  paws::Allocator (block pool)            │
│  Free list, alloc/free, O(1)             │
└─────────────────────────────────────────┘
              │
┌─────────────────────────────────────────┐
│  paws::KVCache (per-sequence)            │
│  Block table, append, free sequence      │
└─────────────────────────────────────────┘
```

```c
// ── paws.h (redesigned) ──

// One-time config from model config.json
typedef struct {
    uint32_t block_tokens;   // tokens per block (default 16)
    uint32_t num_kv_heads;   // GQA-aware
    uint32_t head_dim;       // e.g. 128
    uint32_t precision_bytes;// 2=fp16, 1=int8, 0.5=int4 packed
    uint32_t max_blocks;     // total pool size
} paws_config;

// Single allocator — no per-pool complexity
paws_ctx*     paws_create(const paws_config* cfg);
void          paws_destroy(paws_ctx* ctx);

// ── Block ops (hot path, no locks) ──
int32_t       paws_block_alloc(paws_ctx* ctx);
void          paws_block_free(paws_ctx* ctx, int32_t idx);
void*         paws_block_ptr(paws_ctx* ctx, int32_t idx);  // zero-copy

// ── Sequence ops ──
// Each sequence is a uint32 seq_id (caller manages IDs)
void          paws_seq_init(paws_ctx* ctx, uint32_t seq_id);
int32_t       paws_seq_append(paws_ctx* ctx, uint32_t seq_id);
void          paws_seq_free(paws_ctx* ctx, uint32_t seq_id);

// ── Block table for kernel dispatch ──
// Returns pointer to flat int32 array + length. GPU-readable.
const int32_t* paws_seq_blocks(paws_ctx* ctx, uint32_t seq_id);
uint32_t       paws_seq_len(paws_ctx* ctx, uint32_t seq_id);

// ── Tensor shape for kernel dispatch ──
// The paged attention kernel needs these to stride through blocks.
uint32_t       paws_block_elements(paws_ctx* ctx); // K elements per block
uint32_t       paws_block_bytes(paws_ctx* ctx);    // bytes per block
```

## How the paged attention kernel uses paws

```metal
// Kernel receives from host:
//   K_cache: paws_buffer (one big MTL::Buffer)
//   V_cache: paws_buffer
//   block_table: array of int32 block indices per sequence
//   block_size: constant — tokens per block
//   num_kv_heads, head_dim: constants from model config

for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
    int32_t phys_block = block_table[block_idx];
    uint phys_offset = phys_block * block_elements;
    // Load K_tile from K_cache[phys_offset]
    // Load V_tile from V_cache[phys_offset]
    // Compute attention
}
```

## What makes this scale

| Concern | How paws handles it |
|---------|---------------------|
| New model architecture | Change `paws_config`, no code change |
| GQA | `num_kv_heads` in config, kernel handles replication |
| Sliding window | Caller manages eviction, paws just allocs/frees |
| Mamba2 | `num_kv_heads=0` → paws returns error on seq ops, no crash |
| Different precision | `precision_bytes` in config |
| Continuous batching | Caller manages seq_ids, paws is stateless beyond blocks |
| Multi-GPU | paws is per-device. Two devices = two paws contexts |

## What paws does NOT do

- Does NOT know about model architecture (that's `models/`)
- Does NOT know about tokenization (that's `models/`)
- Does NOT schedule requests (that's `server/`)
- Does NOT manage weight memory (weights are pinned, not allocated)
- Does NOT handle attention patterns (causal/sliding — kernel's job)

paws is a ~200 line C struct. The complexity lives in the paged attention kernel and the model layer above it.
