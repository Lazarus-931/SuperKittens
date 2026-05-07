# Memory on Apple Silicon — what the compiler/kernel author needs

## UMA: One pool, two views

```
┌─────────────────────────────────────────┐
│            Physical RAM (e.g. 16 GB)     │
│                                          │
│  CPU can read/write any byte             │
│  GPU can read/write any byte             │
│  No copies. No transfers. Same memory.   │
└─────────────────────────────────────────┘
```

```c
// This works. No cudaMalloc, no memcpy, no stream sync.
float* data = (float*)malloc(N * sizeof(float));
MTL::Buffer* buf = device->newBufferWithBytesNoCopy(
    data, N * sizeof(float),
    MTL::ResourceStorageModeShared,  // ← "shared" means UMA
    nullptr, nullptr
);
// Dispatch kernel that writes to buf
// data now contains GPU output. Zero copies.
```

## Three storage modes — only one matters

| Mode | What it means | When to use |
|------|--------------|-------------|
| `StorageModeShared` | CPU + GPU see same bytes | **Always** — this is UMA |
| `StorageModePrivate` | GPU-only, CPU can't touch | Never — we want zero-copy |
| `StorageModeManaged` | OS mirrors between CPU/GPU | Never — Apple doesn't need this |

## What "zero-copy" actually means for kernel dispatch

```python
# Our current binding pattern (works, but copies):
x_np = np.random.normal(size=(128, 256)).astype(np.float16)
y_np = np.empty_like(x_np)

# Step 1: Metal allocates new buffer, copies numpy data in
bx = device.newBuffer(x_np.nbytes)    # GPU allocation
memcpy(bx.contents(), x_np.data, ...) # CPU→GPU copy (unnecessary on UMA!)

# Step 2: Dispatch kernel
enc.setBuffer(bx, 0, 0)

# Step 3: Copy output back
memcpy(y_np.data, by.contents(), ...) # GPU→CPU copy (unnecessary on UMA!)
```

The copies in steps 1 and 3 are **waste** on UMA. The fix:

```python
# Zero-copy: numpy array IS the Metal buffer
x_np = np.random.normal(size=(128, 256)).astype(np.float16)
# Make contiguous, aligned, then:
bx = device.newBufferWithBytesNoCopy(
    x_np.ctypes.data, x_np.nbytes,
    MTL::ResourceStorageModeShared, None, None
)
# No memcpy. GPU writes directly to x_np's memory.
```

## What paws adds on top of UMA

paws doesn't change the memory model. It adds **organization**:

```
Before paws:
  app → malloc → MTLBuffer → dispatch
  No structure, no reuse, no quantization awareness

With paws:
  app → paws_alloc(typed, i4_gs64, 4096 elements)
      → returns block index 7
      → block 7: 2048 bytes, dtype=i4, group_size=64
      → scales in block 8
      → GPU reads directly from block 7+8
  No copies. Just organized.
```

## Threadgroup memory — the 32KB hard limit

Every Apple GPU has exactly 32KB of threadgroup (shared) memory per threadgroup. This is the **only** memory constraint the compiler needs to care about.

```
Threadgroup budget for a GEMM tile:
  As[BM][BK] * 2 bytes (fp16)     ← Q tile or A tile
+ Bs[BK][BN] * 2 bytes            ← K tile or B tile  
+ optional: scores, accumulators
─────────────────────────────────
  Must be ≤ 32768 bytes
```

Our `runtime/threadgroup.h` already enforces this at compile time. The DSL tile planner uses this budget to auto-select BM/BN/BK.
