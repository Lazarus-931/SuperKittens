# SK DSL — Apple Silicon Metal compiler

## How it differs from CUTLASS/Triton

| Concern | CUTLASS (CUDA) | Triton (CUDA/ROCm) | SK DSL (Apple Silicon) |
|---------|---------------|---------------------|-------------------------|
| Memory model | Separate host/device, explicit copies | Separate, async copies | **UMA — zero copy, same physical RAM** |
| Threadgroup budget | 48-228KB shared mem | Variable | **32KB fixed** |
| Matmul primitive | Tensor Cores (mma.sync) | Tensor Cores | **simdgroup_multiply_accumulate** |
| Vector width | 128-bit (float4) | 128-bit | **128-bit (half4)** |
| SIMD width | 32 (warp) | 32 | 32 |
| Compiler output | PTX/SASS | PTX/amdgcn | **Metal Shading Language (.metal)** |

## The UMA advantage for the compiler

**Everything is zero-copy.** The compiler never emits `cudaMalloc` or `memcpy`.
A tensor is just a pointer. The CPU writes to it, the GPU reads from it — same bytes.

```c
// No copies in generated code. This just works:
void* buf = malloc(rows * cols * 2);       // CPU-visible
MTL::Buffer* mtl = [device newBufferWithBytesNoCopy:buf ...]; // GPU-visible
// dispatch kernel — GPU writes directly to mtl
// buf now contains GPU output — no memcpy needed
```

This means the DSL doesn't need:
- `to_device()` / `to_host()` calls
- `cudaMemcpy` / `cudaMemcpyAsync`
- Separate allocation for host vs device
- Stream synchronization for memory fences

## Compiler pipeline

```
Python kernel def
      │
      ▼
┌──────────────────┐
│ 1. Tile Planner   │  Given threadgroup budget (32KB), pick BM, BN, BK
│                   │  Auto-select half4 vs scalar load paths
└──────────────────┘
      │
      ▼
┌──────────────────┐
│ 2. Thread Mapper  │  Assign rows to SIMD groups, lanes to columns
│                   │  128 threads, 4 SIMD groups, each lane = 1 half4
└──────────────────┘
      │
      ▼
┌──────────────────┐
│ 3. Barrier Insert │  Insert threadgroup_barrier() at tile boundaries
│                   │  mem_flags::mem_threadgroup for load→compute
│                   │  mem_flags::mem_none for compute→store
└──────────────────┘
      │
      ▼
┌──────────────────┐
│ 4. Codegen        │  Emit .metal file with loops, loads, MMA, stores
│                   │  Write to disk, xcrun metal compiles to .air
└──────────────────┘
      │
      ▼
   kernels/*.metal   ← output, ready to compile
```

## Example: user writes

```python
from sk.dsl import kernel, tile, simd

@kernel(name="gelu_fused")
def gelu(x: Tensor[rows, cols]) -> Tensor[rows, cols]:
    row = simd.group_index()  # which row this SIMD group handles
    off = row * cols
    for k in simd.lane_stride(cols // 4):
        v = x.load_half4(off + k * 4)
        a = 0.044715 * v * v * v
        c = tanh(0.79788456 * (v + a))
        y.store_half4(off + k * 4, 0.5 * v * (1.0 + c))
```

## Generated Metal output

```metal
#include <metal_stdlib>
using namespace metal;

[[host_name("gelu_fused")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void gelu_fused(
    device const half* x [[buffer(0)]],
    device half* y       [[buffer(1)]],
    constant uint& rows  [[buffer(2)]],
    constant uint& cols  [[buffer(3)]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * 4 + simd;
    if (row >= rows) return;
    const size_t off = (size_t)row * cols;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);
    const uint n4 = cols / 4;
    for (uint k = lane; k < n4; k += 32) {
        float4 v = float4(x4[k]);
        float4 a = 0.044715f * v * v * v;
        float4 c = metal::fast::tanh(0.79788456f * (v + a));
        y4[k] = half4(0.5f * v * (1.0f + c));
    }
}
```

## Tile planner — the core algorithm

Given operation type (elementwise, reduction, matmul) and tensor shapes, pick tile sizes that fit in 32KB:

```python
def plan_tiles(op: str, shapes: dict) -> TileConfig:
    budget = 32768  # 32KB
    if op == "matmul":
        # Try largest BM that fits: 2*(BM*BK + BK*BN) * 2 bytes <= 32KB
        for BM in [64, 32, 16, 8]:
            for BN in [64, 32, 16, 8]:
                for BK in [32, 16, 8]:
                    mem = 2 * (BM*BK + BK*BN)  # half = 2 bytes
                    if mem <= budget:
                        return TileConfig(BM, BN, BK)
    elif op == "elementwise":
        return TileConfig(rows=4, cols=32)  # 4 SIMD groups × 32 lanes
```

## What we build now

**Phase 1: Elementwise codegen (today)**
- Supports: gelu, silu, relu, swiglu, rmsnorm — anything with `simd.group_index()` + `simd.lane_stride()`
- Generates valid .metal file
- Verified output matches hand-written kernels

**Phase 2: Tiled matmul (this week)**
- simdgroup MMA wrappers
- Tile planner (BM/BN/BK selection)
- Cooperative load loops with half4 auto-vectorization
- Generates gemm.metal that matches our hand-written perf

**Phase 3: Fused attention (next week)**
- Online softmax in registers
- K/V tile streaming
- Generates attn.metal that matches hand-written
