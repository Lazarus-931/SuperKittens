# paws — memory management for Apple Silicon inference

## What it does

paws is a block allocator on top of MTL::Heap. It manages three pools:

| Pool | Block size | Typical count | Lifetime |
|------|-----------|---------------|----------|
| **Weight pool** | model-defined | 100–500 | Load once, pin forever |
| **KV pool** | 16 tokens × H × d | 1000–10000 | Per-request |
| **Workspace** | Variable | 1–5 | Per-forward-pass |

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    paws                          │
│                                                  │
│  ┌──────────────┐  ┌───────────┐  ┌───────────┐ │
│  │ WeightHeap   │  │  KVHeap   │  │  TempHeap  │ │
│  │ (pin, lookup)│  │(alloc/free│  │(ring buf)  │ │
│  │              │  │  LRU evict│  │            │ │
│  └──────────────┘  └───────────┘  └───────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │              BlockAllocator               │    │
│  │  free_list, block_size, total_blocks      │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │              MTL::Heap (UMA)              │    │
│  │         one allocation, sub-divided       │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

## API

```c
// ── Lifecycle ──
paws_ctx* paws_init(MTL::Device* dev, size_t kv_mb, size_t weight_mb);
void      paws_destroy(paws_ctx* ctx);

// ── Weight pool ──
paws_block* paws_weight_load(paws_ctx* ctx, const char* name,
                              void* data, size_t bytes);
paws_block* paws_weight_lookup(paws_ctx* ctx, const char* name);

// ── KV cache ──
paws_block* paws_kv_alloc(paws_ctx* ctx, uint32_t seq_id);
void        paws_kv_free(paws_ctx* ctx, uint32_t seq_id);
void        paws_kv_append(paws_ctx* ctx, uint32_t seq_id,
                           void* k_data, void* v_data);

// ── KV block table (for paged attention) ──
uint32_t*   paws_kv_block_table(paws_ctx* ctx, uint32_t seq_id);
uint32_t    paws_kv_num_blocks(paws_ctx* ctx, uint32_t seq_id);
```

## KV cache design

```
Physical blocks (fixed 16-token chunks):
  [B0] [B1] [B2] ... [BN]     ← MTL::Heap

Sequence 0:  B0 → B4 → B7     ← linked via block_table
Sequence 1:  B1 → B3 → B5 → B9
Sequence 2:  (none yet)

block_table[seq_id] = array of physical block indices
Free list: doubly-linked in-place (uses block metadata headers)
```

## Dev plan

### Phase 1: Block allocator (day 1)
- `BlockAllocator` C++ class: free list, allocate, free, stats
- Test: allocate 1000 blocks, free random 500, allocate 500 more, verify no leaks
- Test: allocate until exhaustion, verify returns NULL
- File: `paws/allocator.h`, `paws/allocator.c++`, `paws/test_allocator.c++`

### Phase 2: KV cache (day 2)
- `KVHeap`: block allocator + per-sequence block table
- `kv_alloc(seq_id)`: get one free block, append to sequence
- `kv_free(seq_id)`: return all blocks for a sequence to free list
- `kv_block_table(seq_id)`: return flat array of block indices
- Test: 10 sequences, random appends, read back block tables, verify no overlap
- Files: `paws/cache.h`, `paws/cache.c++`, `paws/test_cache.c++`

### Phase 3: Weight pool (day 2–3)
- `WeightHeap`: name → block mapping, reference counting
- `weight_load(name, data, bytes)`: allocate, copy, register
- `weight_lookup(name)`: find existing
- Test: load 200 tensors, lookup all, verify shapes
- Files: `paws/weights.h`, `paws/weights.c++`

### Phase 4: Integration test (day 3)
- Dummy model that allocates KV blocks, loads weights, runs kernel stubs
- `paws/integration.c++`: full lifecycle — init → load → allocate → free → destroy
- Bench: allocation latency, fragmentation over 100K ops

### Phase 5: Python bindings (day 3–4)
- `paws/paws.h`: unified C API
- `paws/paws.c++`: single .c++ wrapping all three subsystems
- `paws/paws.py`: ctypes bindings
- Test: `python3 paws/test.py` — create cache, alloc/free 1000 blocks, verify

## How to test

```bash
# C++ unit test (no Metal needed)
clang++ paws/test_allocator.c++ paws/allocator.c++ -o build/test_allocator
./build/test_allocator

# Metal integration test
clang++ paws/test_cache.c++ paws/cache.c++ paws/allocator.c++ \
    -framework Metal -framework Foundation -o build/test_cache
./build/test_cache

# Python test
python3 paws/test.py
```

## What we can accomplish

| Milestone | What runs |
|-----------|-----------|
| Phase 1 done | Block allocator works |
| Phase 2 done | Paged KV cache works, ready for paged attention kernel |
| Phase 3 done | Full model weight loading works |
| Phase 4 done | End-to-end: load weights → allocate KV → run attention → free |
| Phase 5 done | Python can manage KV cache for continuous batching |

paws + our existing kernel bindings + a tokenizer = can serve a real model.
