//
//  ple_inject.metal — Gemma 4 Per-Layer-Embedding pipeline.
//
//  Three kernels:
//    gemma4_ple_lookup       : ids → per_layer_inputs[T, n_layers, PLE_dim]
//    gemma4_ple_gate_act     : out[T, PLE_dim] = gelu_approx(gate[T, PLE_dim]) * ple_slice[L]
//    gemma4_ple_inject       : residual += scalar[L] * rmsnorm(proj_back, gamma[L])
//

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::ple_inject {

[[host_name("gemma4_ple_lookup")]]
[[kernel]]
void gemma4_ple_lookup(
    device const half* ple_table       [[buffer(0)]],   // (V, n_layers, P)
    device const int*  ids             [[buffer(1)]],   // (T,)
    device       half* per_layer_inputs[[buffer(2)]],   // (T, n_layers, P)
    constant uint& T                   [[buffer(3)]],
    constant uint& n_layers            [[buffer(4)]],
    constant uint& P                   [[buffer(5)]],
    constant uint& V                   [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint t = gid.z;
    const uint L = gid.y;
    const uint p4 = gid.x;
    const uint P4 = P / 4u;
    if (t >= T || L >= n_layers || p4 >= P4) return;

    int  id  = ids[t];
    bool oob = (id < 0) || ((uint)id >= V);
    half4 v = half4(0, 0, 0, 0);
    if (!oob) {
        const uint row_off = ((uint)id * n_layers + L) * P;
        v = reinterpret_cast<const device half4*>(ple_table + row_off)[p4];
    }
    const uint dst_off = (t * n_layers + L) * P;
    reinterpret_cast<device half4*>(per_layer_inputs + dst_off)[p4] = v;
}

static inline half gelu_approx(half x) {
    const half c0 = (half)0.7978845608028654h;     // sqrt(2/pi)
    const half c1 = (half)0.044715h;
    half x3 = x * x * x;
    half u  = c0 * (x + c1 * x3);
    half t  = tanh(u);
    return (half)0.5h * x * ((half)1.0h + t);
}

[[host_name("gemma4_ple_gate_act")]]
[[kernel]]
void gemma4_ple_gate_act(
    device const half* gate            [[buffer(0)]],   // (T, P)
    device const half* per_layer_inputs[[buffer(1)]],   // (T, n_layers, P)
    device       half* out             [[buffer(2)]],   // (T, P)
    constant uint& T                   [[buffer(3)]],
    constant uint& n_layers            [[buffer(4)]],
    constant uint& P                   [[buffer(5)]],
    constant uint& layer_idx           [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint t  = gid.y;
    const uint p4 = gid.x;
    const uint P4 = P / 4u;
    if (t >= T || p4 >= P4) return;

    half4 g  = reinterpret_cast<const device half4*>(gate + t * P)[p4];
    const uint ple_off = (t * n_layers + layer_idx) * P;
    half4 ple= reinterpret_cast<const device half4*>(per_layer_inputs + ple_off)[p4];

    half4 a;
    a.x = gelu_approx(g.x) * ple.x;
    a.y = gelu_approx(g.y) * ple.y;
    a.z = gelu_approx(g.z) * ple.z;
    a.w = gelu_approx(g.w) * ple.w;
    reinterpret_cast<device half4*>(out + t * P)[p4] = a;
}

[[host_name("gemma4_ple_inject")]]
[[kernel]]
void gemma4_ple_inject(
    device const half*  proj_back  [[buffer(0)]],   // (T, D)
    device const half*  gamma      [[buffer(1)]],   // (D,) for this layer
    device       half*  residual   [[buffer(2)]],   // (T, D) in-place
    constant uint&  T              [[buffer(3)]],
    constant uint&  D              [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    device const float* scalar_arr [[buffer(6)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  tgs [[threads_per_threadgroup]])
{
    if (row >= T) return;

    device const half* xrow = proj_back + (size_t)row * D;
    device       half* rrow = residual  + (size_t)row * D;

    threadgroup float tg_sum[32];

    float local = 0.0f;
    for (uint i = tid; i < D; i += tgs) {
        float v = (float)xrow[i];
        local += v * v;
    }
    const uint lane    = tid & 31u;
    const uint warp_id = tid >> 5;
    float wsum = local;
    wsum += simd_shuffle_xor(wsum, 16);
    wsum += simd_shuffle_xor(wsum,  8);
    wsum += simd_shuffle_xor(wsum,  4);
    wsum += simd_shuffle_xor(wsum,  2);
    wsum += simd_shuffle_xor(wsum,  1);
    if (lane == 0) tg_sum[warp_id] = wsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    const uint n_warps = (tgs + 31u) >> 5;
    if (tid < n_warps) total = tg_sum[tid];
    total += simd_shuffle_xor(total, 16);
    total += simd_shuffle_xor(total,  8);
    total += simd_shuffle_xor(total,  4);
    total += simd_shuffle_xor(total,  2);
    total += simd_shuffle_xor(total,  1);
    if (tid == 0) tg_sum[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean   = tg_sum[0] / (float)D;
    float invrms = rsqrt(mean + eps);
    float scl    = scalar_arr[0];

    for (uint i = tid; i < D; i += tgs) {
        float n = (float)xrow[i] * invrms * (float)gamma[i];
        float r = ((float)rrow[i] + n) * scl;
        rrow[i] = (half)r;
    }
}

} // namespace meow::gemma4::ple_inject
