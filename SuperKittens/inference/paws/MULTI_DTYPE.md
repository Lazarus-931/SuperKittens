# paws — multi-dtype memory design

## paws doesn't need to know

paws is a block allocator. It manages bytes, not tensors. The dtype lives
one layer up, in the runtime tensor descriptor.

```
paws:       "here's 4096 bytes at block 7"
runtime:    "block 7 is (64, 32) int4 with scales in block 8"
kernel:     "load uchar, dequant with scales, compute fp16"
```

## Where each concern lives

| Layer | Knows about | Example |
|-------|------------|---------|
| `paws` | Bytes, free list, block table | `paws_block_alloc(ctx)` |
| `runtime/tensor.h` | Shape, dtype, scales ptr | `Tensor { .shape={64,32}, .dtype=INT4, .scales=block_8 }` |
| `kernels/gemm` | How to load + dequant | `gemm_int4.metal` |
| `models/llama.py` | Which layers are which dtype | Linear: int4, Norm: fp16 |

## A tensor in the system

```c
// runtime/tensor.h — the dtype-aware layer above paws
typedef struct {
    uint32_t shape[4];
    uint8_t  ndim;

    // dtype + quantization metadata
    enum { SK_F16, SK_F32, SK_I8, SK_I4, SK_I4_GS64 } dtype;

    // For quantized tensors: per-block scales
    uint32_t group_size;   // 64 or 128 (0 = per-channel)
    int32_t  scales_block; // paws block index for scales
    int32_t  zeros_block;  // paws block index for zero points (optional)

    // The data itself
    int32_t  data_block;   // paws block index
    size_t   data_bytes;   // total bytes in this block
} SKTensor;
```

## How dispatch picks the right kernel

```python
# kernels/gemm/gemm.py
def gemm(A: SKTensor, B: SKTensor, C: SKTensor):
    # Dispatch table
    kernel_map = {
        (SK_F16, SK_F16): "gemm_fp16",
        (SK_I4,  SK_F16): "gemm_i4_f16",
        (SK_I8,  SK_F16): "gemm_i8_f16",
    }
    name = kernel_map[(A.dtype, B.dtype)]
    # paws_block_ptr resolves data pointers
    lib.sk_gemm_dispatch(name.encode(),
                         paws_block_ptr(A.data_block), paws_block_ptr(A.scales_block),
                         paws_block_ptr(B.data_block), paws_block_ptr(B.scales_block),
                         ...)
```

## What paws adds for multi-dtype

Three things, all in `paws.h`:

```c
// 1. Per-block metadata (8 bytes per block, stored inline)
typedef struct {
    uint8_t  dtype;       // SK_F16, SK_I4, etc.
    uint8_t  flags;       // has_scales, has_zeros
    uint16_t group_size;  // 0 = no quantization
    uint32_t _reserved;
} paws_block_meta;

// Set when block is allocated for a specific dtype
void paws_block_set_meta(ctx, block_idx, dtype, group_size);

// 2. Allocate block with metadata
int32_t paws_block_alloc_typed(ctx, dtype, group_size);

// 3. Query — zero cost, just reads the header
const paws_block_meta* paws_block_meta(ctx, block_idx);
```

The metadata lives in the first 8 bytes of each block. Free blocks use
those 8 bytes for the free list pointer. Allocated blocks use them for
dtype metadata. Same bytes, different interpretation.

## Model loading flow

```python
# models/loader.py
def load_safetensors(path, paws_ctx):
    for name, tensor in safetensors.safe_open(path):
        # Parse dtype from safetensors metadata
        if "int4" in name or tensor is quantized:
            dtype = SK_I4
            group_size = 64
            data_blk = paws_block_alloc_typed(ctx, SK_I4, group_size)
            scales_blk = paws_block_alloc_typed(ctx, SK_F16, 0)
        else:
            dtype = SK_F16
            data_blk = paws_block_alloc_typed(ctx, SK_F16, 0)
            scales_blk = -1

        # Copy raw bytes into the block
        memcpy(paws_block_ptr(ctx, data_blk), tensor_data, bytes)
        if scales_blk >= 0:
            memcpy(paws_block_ptr(ctx, scales_blk), tensor_scales, scales_bytes)

        # Register in the tensor table
        tensors[name] = SKTensor(data_blk, shape, dtype, scales_blk)
```

## Why this is clean

- paws stays simple — 8 extra bytes per block for metadata, no new allocators
- The runtime tensor descriptor knows dtype but NOT Metal — just shape + dtype + block indices
- The kernel dispatch table maps (dtype_A, dtype_B) → kernel name
- Models specify which layers are quantized, same as PyTorch
- Zero-copy unchanged — still UMA, paws just gives pointers
