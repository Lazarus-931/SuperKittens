# paws — status & missing pieces

## Done

| Component | File | Status |
|-----------|------|--------|
| Block allocator (free list, alloc/free) | `paws.h/c++` | ✓ compiled, tested |
| Per-sequence block tables | `paws.h/c++` | ✓ compiled, tested |
| DType enum (10 types) | `dtype.h/c++` | ✓ compiled, tested |
| Chip detection (M1-M4) | `dtype.h/c++` | ✓ compiled, tested |
| Chip-aware GEMM dispatch table | `dtype.c++` | ✓ compiled, tested |
| DType compatibility check | `dtype.h` | ✓ compiled, tested |

## Missing

### P0: Quantizer — `paws/quantize.c++`

fp16 weights → int4 packed + scales. This is what makes quantization real.

```c
// Signature
int paws_quantize_fp16_to_i4(
    paws_ctx* ctx,
    const void* fp16_weights,   // input: fp16 data
    uint32_t rows, uint32_t cols,
    uint32_t group_size,         // 64 or 128
    int32_t* out_data_block,     // output: paws block index for packed int4
    int32_t* out_scales_block    // output: paws block index for fp16 scales
);
```

Implementation: ~50 lines. Loop over groups, compute per-group scale, pack 2 int4 per byte.

### P0: `paws_block_alloc_typed()` — allocate with metadata

```c
// Allocate a block and tag it with dtype + group_size
int32_t paws_block_alloc_typed(paws_ctx* ctx, paws_dtype_t dtype, uint16_t group_size);
```

Stores 8-byte metadata header in the first bytes of the block. Free blocks reuse those bytes for the free list pointer. Same memory, different interpretation.

### P1: Zero-copy numpy integration

```python
# paws/paws.py
class PawsBuffer:
    """A numpy array that IS a Metal buffer — zero copy."""
    def __init__(self, shape, dtype):
        self.np = np.empty(shape, dtype=dtype)
        self.mtl = device.newBufferWithBytesNoCopy(
            self.np.ctypes.data, self.np.nbytes,
            MTL.ResourceStorageModeShared, None, None
        )
```

### P1: `paws_block_meta()` — read dtype from a block

```c
typedef struct { uint8_t dtype; uint8_t flags; uint16_t group_size; } paws_block_meta;
const paws_block_meta* paws_block_meta(paws_ctx* ctx, int32_t block_idx);
```

### P2: Block compaction/defrag

When the free list gets fragmented, compact used blocks to create contiguous free space. Important for long-running inference servers.

### P2: Python bindings for paws

ctypes wrapper for `paws.h` + `dtype.h`. Enables Python model loaders to use paws directly.

## What paws should NOT do

- Model architecture (that's `models/`)
- Tokenization
- Request scheduling (that's `server/`)
- Kernel dispatch (that's the bindings)
- Weight file format parsing (that's `models/loader.py`)
