//
//  ple_inject_fused.metal — single-dispatch decode-time PLE inject.
//
//  Replaces the 4-dispatch pipeline (gate matvec → gelu·ple → proj matvec →
//  rmsnorm+add) with a single threadgroup-resident kernel for T=1 / d_model
//  ≤ 2048. Each threadgroup processes one (token,layer) and stages all
//  intermediates in shared memory, eliminating 4 encoder switches and
//  3 device-memory round trips per layer.
//
//  Decode-only fast path. Prefill (T>1) still uses the unfused pipeline
//  because gate[P] and proj[D] would exceed reasonable TG memory for large T.
//

#include <metal_stdlib>
using namespace metal;

namespace meow::gemma4::ple_fused {

// Tunables: GEMMA4 E2B has d_model=1536, ple_dim=256.
// E4B has d_model=2560 — we cap D_MAX at 2560 so both fit. TG memory budget:
//   residual_tg : D_MAX * 2 bytes (bf16)   = 5120 B
//   gate_tg     : P_MAX * 4 bytes (fp32)   = 1024 B
//   proj_tg     : D_MAX * 4 bytes (fp32)   = 10240 B
//   reduce_tg   : 8 * 4                    = 32 B
//   total                                  ≈ 16 KB — well under Apple's 32 KB limit.
enum : uint { D_MAX = 2560u, P_MAX = 256u, TG_THREADS = 256u };

static inline float gelu_approx_f(float x) {
    const float c0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float c1 = 0.044715f;
    float x3 = x * x * x;
    float u  = c0 * (x + c1 * x3);
    float t  = metal::precise::tanh(u);
    return 0.5f * x * (1.0f + t);
}

// One threadgroup = one (token, layer). For decode this is 1 dispatch with
// 1 threadgroup per layer (vs. the previous 4 dispatches per layer).
//
// Buffers:
//   0  residual           (T, D) bf16, in-place
//   1  W_gate slice       (D, P) bf16, offset baked by caller (= L*D*P*2 bytes)
//   2  W_proj slice       (P, D) bf16, offset baked by caller (= L*P*D*2 bytes)
//   3  W_post_norm slice  (D,)   bf16, offset baked by caller (= L*D*2 bytes)
//   4  per_layer_inputs   (T, n_layers, P) bf16
//   5  scalar_arr         (n_layers,) fp32, offset baked by caller (= L*4 bytes)
//   6  uint D
//   7  uint P
//   8  uint n_layers
//   9  uint layer_idx
//  10  float eps
[[host_name("gemma4_ple_inject_fused_t1")]]
[[kernel]]
void gemma4_ple_inject_fused_t1(
    device       bfloat*  residual         [[buffer(0)]],
    device const bfloat*  w_gate           [[buffer(1)]],
    device const bfloat*  w_proj           [[buffer(2)]],
    device const bfloat*  w_post_norm      [[buffer(3)]],
    device const bfloat*  per_layer_inputs [[buffer(4)]],
    device const float*   scalar_arr       [[buffer(5)]],
    constant uint&  D          [[buffer(6)]],
    constant uint&  P          [[buffer(7)]],
    constant uint&  n_layers   [[buffer(8)]],
    constant uint&  layer_idx  [[buffer(9)]],
    constant float& eps        [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    // Shared staging.
    threadgroup bfloat residual_tg[D_MAX];
    threadgroup float  gate_tg[P_MAX];
    threadgroup float  proj_tg[D_MAX];
    threadgroup float  reduce_tg[8];

    // ---- Load residual into TG (bf16) -------------------------------------
    for (uint i = tid; i < D; i += tgs) {
        residual_tg[i] = residual[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 1: gate[p] = sum_d residual[d] * W_gate[d*P + p] -----------
    // One thread per gate output (P outputs, P ≤ TG_THREADS).
    if (tid < P) {
        float acc = 0.0f;
        for (uint d = 0; d < D; ++d) {
            acc += float(residual_tg[d]) * float(w_gate[d * P + tid]);
        }
        gate_tg[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 2: gated[p] = gelu(gate[p]) * per_layer_inputs[L, p] -------
    if (tid < P) {
        const uint ple_off = layer_idx * P + tid;   // T=1 so token offset = 0
        float g  = gate_tg[tid];
        float pi = float(per_layer_inputs[ple_off]);
        gate_tg[tid] = gelu_approx_f(g) * pi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Step 3: proj[d] = sum_p gated[p] * W_proj[p*D + d] --------------
    // Each thread handles a strided chunk of D outputs.
    float local_sumsq = 0.0f;
    for (uint d = tid; d < D; d += tgs) {
        float acc = 0.0f;
        for (uint p = 0; p < P; ++p) {
            acc += gate_tg[p] * float(w_proj[p * D + d]);
        }
        proj_tg[d] = acc;
        local_sumsq += acc * acc;
    }

    // ---- Step 4: rmsnorm(proj) * gamma + residual, all scaled -----------
    // Reduce sum-of-squares across the threadgroup.
    const uint lane   = tid & 31u;
    const uint warp   = tid >> 5;
    float wsum = local_sumsq;
    wsum += simd_shuffle_xor(wsum, 16);
    wsum += simd_shuffle_xor(wsum,  8);
    wsum += simd_shuffle_xor(wsum,  4);
    wsum += simd_shuffle_xor(wsum,  2);
    wsum += simd_shuffle_xor(wsum,  1);
    if (lane == 0) reduce_tg[warp] = wsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    const uint n_warps = (tgs + 31u) >> 5;
    if (tid < n_warps) total = reduce_tg[tid];
    total += simd_shuffle_xor(total, 16);
    total += simd_shuffle_xor(total,  8);
    total += simd_shuffle_xor(total,  4);
    total += simd_shuffle_xor(total,  2);
    total += simd_shuffle_xor(total,  1);
    if (tid == 0) reduce_tg[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float invrms = rsqrt(reduce_tg[0] / float(D) + eps);
    const float scl    = scalar_arr[0];

    for (uint d = tid; d < D; d += tgs) {
        float n = proj_tg[d] * invrms * float(w_post_norm[d]);
        float r = (float(residual_tg[d]) + n) * scl;
        residual[d] = bfloat(r);
    }
}

} // namespace meow::gemma4::ple_fused
