# Quantization on Apple Silicon — what you need to know

## The only rule that matters

Apple GPU has **no native int4/int8 matmul hardware**. All quantized matmul
works the same way: load compressed weights, dequant to fp16, feed into the
same fp16 simdgroup MMA.

```
On NVIDIA:    int4 → Tensor Core → accumulate in int32 → convert
On Apple:     int4 → dequant to fp16 → simdgroup MMA in fp16 → no conversion
```

This means quantization on Apple saves **memory bandwidth**, not compute.
The compute is always fp16.

## Weight-only quantization (what inference uses)

Llama 7B with fp16: 14 GB weights. With int4: 3.5 GB weights + scales.
The 4× memory reduction means a 7B model that didn't fit now fits.

### Per-channel (simplest)

```python
# One scale per output channel. No zero point.
scale = max(abs(weight[row, :])) / 7.0    # int4 range [-8,7]
weight_i4[row, :] = round(weight[row, :] / scale[row])

# Dequant at runtime:
weight_fp16[row, :] = weight_i4[row, :] * scale[row]
```

### Per-group (better accuracy, standard)

```python
# One scale per group_size elements within a channel.
# group_size = 64 or 128. Smaller = more accurate, more scales.
for g in range(0, K, group_size):
    scale[g//group_size] = max(abs(weight[:, g:g+group_size])) / 7.0
    weight_i4[:, g:g+group_size] = round(weight[:, g:g+group_size] / scale)
```

### Which layers to quantize

| Layer | Quantize? | Why |
|-------|-----------|-----|
| Q/K/V projections | Yes | 4× GEMM weights = most of the model |
| FFN up/gate/down | Yes | Same — heavy matmuls |
| Output projection | Yes | |
| Embedding | Yes (int8) | int4 hurts embedding quality |
| RMSNorm weights | No | Tiny (4096 elements), no benefit |
| Attention scores | No | Computed, not stored |
| KV cache | Optional (int8) | Saves cache memory, minimal accuracy loss |

## How the dequant kernel works

```metal
// In the GEMM As load loop:
device const uchar* packed = (device const uchar*)weight_i4;
for (uint i = 0; i < As_elems; i += 2) {
    uchar pair = packed[i / 2];
    // Unpack 2 int4 weights from 1 byte
    float w0 = (float)((pair & 0x0F) - 8) * scale[g];  // lower nibble
    float w1 = (float)((pair >> 4) - 8) * scale[g];     // upper nibble
    As[i]   = half(w0);
    As[i+1] = half(w1);
}
```

This adds ~5-8% overhead vs fp16 GEMM. The scales are pre-loaded into
registers. For group_size=64, one scale per 64 weights — negligible overhead.

## What the paw quantizer does

```
fp16 weights (numpy/safetensors)
    │
    ▼ paws_quantize_fp16_to_i4()
    │
    ├── data_block   → paws block (int4 packed, 2 per byte)
    ├── scales_block → paws block (fp16, one per group)
    └── metadata     → dtype=I4_GS64, group_size=64
```

The quantizer is offline — run once at model load time. The dequant is
runtime — runs inside the GEMM kernel every forward pass.

## Mixed precision dispatch (already built)

```python
# models/llama.py
A_dtype = PAWS_I4_GS64   # weight is int4 quantized
B_dtype = PAWS_F16       # activation stays fp16

kernel_name = paws_gemm_dtype_name(A_dtype, B_dtype)
# → "gemm_i4_f16"  (or "gemm_fp16" as fallback on older chips)

# The kernel handles the rest:
# - loads packed int4 weights
# - loads fp16 scales
# - dequants on-the-fly in the load loop
# - feeds fp16 values into the standard MMA
```

## What we have vs what's missing

| Component | Status |
|-----------|--------|
| Dtype system (I4, I4_GS64, etc.) | ✓ paws/dtype.h |
| Chip-aware dispatch for quantized GEMM | ✓ paws/dtype.c++ |
| Block allocator for data + scales | ✓ paws/paws.h |
| int4 GEMM kernel | ✗ Need to write gemm_i4_f16.metal |
| Quantizer (fp16 → int4) | ✗ Need to write paws/quantize.c++ |
| Runtime tensor struct | ✗ Need SKTensor with dtype + scales ptr |
