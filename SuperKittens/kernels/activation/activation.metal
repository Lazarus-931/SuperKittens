//
//  activation.metal — per-chip register-aware tile sizing
//
//  M3+ (Apple 9/10): 256 threads, 8 rows/group (128 regs → larger tiles)
//  M1/M2 (Apple 7/8): 128 threads, 4 rows/group (96/112 regs)
//

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, SIMDS = 4, ROWS_PER_GROUP = 4 };

// ── GELU: fast::tanh — 3x faster than precise, negligible accuracy loss ──

[[host_name("gelu")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void gelu(
    device const half* x, device half* y,
    constant uint& rows, constant uint& cols,
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GROUP + simd;
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
    for (uint k = n4 * 4 + lane; k < cols; k += 32) {
        float v = float(x[off + k]);
        float a = 0.044715f * v * v * v;
        float c = metal::fast::tanh(0.79788456f * (v + a));
        y[off + k] = half(0.5f * v * (1.0f + c));
    }
}

// ── SiLU ───────────────────────────────────────────────────────────

[[host_name("silu")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void silu(
    device const half* x, device half* y,
    constant uint& rows, constant uint& cols,
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GROUP + simd;
    if (row >= rows) return;
    const size_t off = (size_t)row * cols;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);
    const uint n4 = cols / 4;

    for (uint k = lane; k < n4; k += 32) {
        float4 v = float4(x4[k]);
        y4[k] = half4(v / (1.0f + metal::fast::exp(-v)));
    }
    for (uint k = n4 * 4 + lane; k < cols; k += 32) {
        float v = float(x[off + k]);
        y[off + k] = half(v / (1.0f + metal::fast::exp(-v)));
    }
}

// ── ReLU ───────────────────────────────────────────────────────────

[[host_name("relu")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void relu(
    device const half* x, device half* y,
    constant uint& rows, constant uint& cols,
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint row = gid.y * ROWS_PER_GROUP + simd;
    if (row >= rows) return;
    const size_t off = (size_t)row * cols;
    const device half4* x4 = reinterpret_cast<const device half4*>(x + off);
    device half4* y4 = reinterpret_cast<device half4*>(y + off);
    const uint n4 = cols / 4;

    for (uint k = lane; k < n4; k += 32) {
        y4[k] = half4(fmax(float4(x4[k]), 0.0f));
    }
    for (uint k = n4 * 4 + lane; k < cols; k += 32) {
        y[off + k] = half(fmax(float(x[off + k]), 0.0f));
    }
}
