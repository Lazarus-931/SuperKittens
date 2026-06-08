//
//  ple_inject.metal — Gemma 4 Per-Layer-Embedding pipeline.
//
//  Three kernels:
//    gemma4_ple_lookup       : ids → per_layer_inputs[T, n_layers, PLE_dim]
//    gemma4_ple_gate_act     : out[T, PLE_dim] = gelu_approx(gate[T, PLE_dim]) * ple_slice[L]
//    gemma4_ple_inject       : residual += scalar[L] * rmsnorm(proj_back, gamma[L])
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

namespace meow::gemma4::ple_inject {

// ──────────────────────────────────────────────────────────────────────
//  gemma4_gemm_bf16_fp32_out
//  Same as gemma4_gemm_bf16 but C is fp32. Used by the PLE-inject path
//  for the two small GEMMs (T,d_model)@(d_model,P) and (T,P)@(P,d_model)
//  so the intermediate `ple_gate_out` and `ple_proj_back` keep full fp32
//  precision (avoids 3× bf16 quantization round-trips that get amplified
//  by L1's input_layernorm gamma_max=76).
// ──────────────────────────────────────────────────────────────────────
[[host_name("gemma4_gemm_bf16_fp32_out")]]
[[kernel]]
void gemma4_gemm_bf16_fp32_out(
    device const bfloat* A     [[buffer(0)]],
    device const bfloat* B     [[buffer(1)]],
    device       float*  C     [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    constant uint& ldA       [[buffer(6)]],
    constant uint& ldB       [[buffer(7)]],
    constant uint& ldC       [[buffer(8)]],
    constant bool& transA    [[buffer(9)]],
    constant bool& transB    [[buffer(10)]],
    constant bool& has_bias  [[buffer(11)]],
    device const bfloat* bias [[buffer(12)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    enum : uint { BM = 32, BN = 64, BK = 32, MR = 2, MC = 8 };
    const uint br = gid.y * BM, bc = gid.x * BN;
    const uint r0 = simd * MR * 8;

    threadgroup bfloat As[BM * BK];
    threadgroup bfloat Bs[BK * BN];
    simdgroup_float8x8 acc[MR][MC] = {};

    const uint n_a = BM * BK, n_b = BK * BN;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        if (!transA) {
            for (uint i = simd * 32 + lane; i < n_a / 4; i += 64) {
                uint r = (i * 4) / BK, c = (i * 4) % BK;
                uint gr = br + r, gc = k0 + c;
                reinterpret_cast<threadgroup bfloat4*>(As)[i] =
                    (gr < M && gc < K) ? reinterpret_cast<const device bfloat4*>(A + gr * ldA + gc)[0] : bfloat4(0);
            }
        } else {
            for (uint i = simd * 32 + lane; i < n_a; i += 64) {
                uint r = i / BK, c = i % BK;
                uint gr = k0 + c, gc = br + r;
                As[i] = (gr < K && gc < M) ? A[gr * ldA + gc] : bfloat(0);
            }
        }
        if (!transB) {
            for (uint i = simd * 32 + lane; i < n_b / 4; i += 64) {
                uint r = (i * 4) / BN, c = (i * 4) % BN;
                uint gr = k0 + r, gc = bc + c;
                reinterpret_cast<threadgroup bfloat4*>(Bs)[i] =
                    (gr < K && gc < N) ? reinterpret_cast<const device bfloat4*>(B + gr * ldB + gc)[0] : bfloat4(0);
            }
        } else {
            for (uint i = simd * 32 + lane; i < n_b; i += 64) {
                uint r = i / BN, c = i % BN;
                uint gr = bc + c, gc = k0 + r;
                Bs[i] = (gr < N && gc < K) ? B[gr * ldB + gc] : bfloat(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK / 8; k++) {
            for (uint r = 0; r < MR; r++) {
                simdgroup_bfloat8x8 a;
                simdgroup_load(a, As + (r0 + r * 8) * BK + k * 8, BK);
                for (uint c = 0; c < MC; c++) {
                    simdgroup_bfloat8x8 b;
                    simdgroup_load(b, Bs + k * 8 * BN + c * 8, BN);
                    simdgroup_multiply_accumulate(acc[r][c], a, b, acc[r][c]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulator → tg → device (fp32).
    threadgroup float Cs[BM * BN];
    for (uint r = 0; r < MR; r++) {
        for (uint c = 0; c < MC; c++) {
            float2 v = reinterpret_cast<thread float2&>(acc[r][c].thread_elements());
            uint qid = lane / 4;
            uint lr = (qid & 4) + ((lane / 2) % 4);
            uint lc = (qid & 2) * 2 + (lane % 2) * 2;
            Cs[(r0 + r * 8 + lr) * BN + c * 8 + lc]     = v.x;
            Cs[(r0 + r * 8 + lr) * BN + c * 8 + lc + 1] = v.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = simd * 32 + lane; i < (BM * BN) / 4; i += 64) {
        uint r = (i * 4) / BN, c = (i * 4) % BN;
        uint gr = br + r, gc = bc + c;
        if (gr < M && gc < N) {
            float4 val = reinterpret_cast<threadgroup float4*>(Cs)[i];
            if (has_bias) {
                bfloat4 b4 = reinterpret_cast<const device bfloat4*>(bias + gc)[0];
                val = val + float4(b4);
            }
            reinterpret_cast<device float4*>(C + gr * ldC + gc)[0] = val;
        }
    }
}

[[host_name("gemma4_ple_lookup")]]
[[kernel]]
void gemma4_ple_lookup(
    device const bfloat* ple_table       [[buffer(0)]],   // (V, n_layers, P)
    device const int*  ids             [[buffer(1)]],   // (T,)
    device       bfloat* per_layer_inputs[[buffer(2)]],   // (T, n_layers, P)
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
    bfloat4 v = bfloat4(0, 0, 0, 0);
    if (!oob) {
        const uint row_off = ((uint)id * n_layers + L) * P;
        v = reinterpret_cast<const device bfloat4*>(ple_table + row_off)[p4];
    }
    const uint dst_off = (t * n_layers + L) * P;
    reinterpret_cast<device bfloat4*>(per_layer_inputs + dst_off)[p4] = v;
}

// Q8_0 block: half d (per-32 scale) + int8 qs[32] → 34 bytes / 32 weights.
// Matches kernels/gemm/q8_0_matvec_bf16.metal's q8b_block.
struct __attribute__((packed)) g4_q8_block { half d; int8_t qs[32]; };

// gemma4_ple_lookup_q8 — Q8_0 PLE-table gather + dequant.
// PLE table stored Q8_0 row-major (V, n_layers*P); each row is gathered for
// token id and dequantized into per_layer_inputs[T, n_layers, P] (bf16). One
// thread per output element (no 4-vec: Q8 dequant is per-element).
[[host_name("gemma4_ple_lookup_q8")]]
[[kernel]]
void gemma4_ple_lookup_q8(
    device const uchar*  ple_table_q8     [[buffer(0)]],   // (V, n_layers*P) q8_0
    device const int*    ids              [[buffer(1)]],   // (T,)
    device       bfloat* per_layer_inputs [[buffer(2)]],   // (T, n_layers, P)
    constant uint& T                      [[buffer(3)]],
    constant uint& n_layers               [[buffer(4)]],
    constant uint& P                      [[buffer(5)]],
    constant uint& V                      [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint t = gid.z;
    const uint L = gid.y;
    const uint p = gid.x;
    if (t >= T || L >= n_layers || p >= P) return;

    const uint row_len = n_layers * P;          // elems per token row
    const uint col     = L * P + p;              // elem index within the row
    int  id  = ids[t];
    bool oob = (id < 0) || ((uint)id >= V);

    bfloat val = bfloat(0);
    if (!oob) {
        const uint nb_row = row_len / 32u;       // blocks per token row
        device const g4_q8_block* row =
            (device const g4_q8_block*)(ple_table_q8) + (size_t)id * nb_row;
        const uint blk = col >> 5;               // /32
        const uint lane = col & 31u;
        val = bfloat((float)row[blk].qs[lane] * (float)row[blk].d);
    }
    const uint dst = (t * n_layers + L) * P + p;
    per_layer_inputs[dst] = val;
}

static inline float gelu_approx(float x) {
    const float c0 = 0.7978845608028654f;     // sqrt(2/pi)
    const float c1 = 0.044715f;
    float x3 = x * x * x;
    float u  = c0 * (x + c1 * x3);
    float t  = metal::precise::tanh(u);
    return 0.5f * x * (1.0f + t);
}

[[host_name("gemma4_ple_gate_act")]]
[[kernel]]
void gemma4_ple_gate_act(
    device const bfloat* gate            [[buffer(0)]],   // (T, P)
    device const bfloat* per_layer_inputs[[buffer(1)]],   // (T, n_layers, P)
    device       bfloat* out             [[buffer(2)]],   // (T, P)
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

    bfloat4 g  = reinterpret_cast<const device bfloat4*>(gate + t * P)[p4];
    const uint ple_off = (t * n_layers + layer_idx) * P;
    bfloat4 ple= reinterpret_cast<const device bfloat4*>(per_layer_inputs + ple_off)[p4];

    bfloat4 a;
    a.x = bfloat(gelu_approx(float(g.x)) * float(ple.x));
    a.y = bfloat(gelu_approx(float(g.y)) * float(ple.y));
    a.z = bfloat(gelu_approx(float(g.z)) * float(ple.z));
    a.w = bfloat(gelu_approx(float(g.w)) * float(ple.w));
    reinterpret_cast<device bfloat4*>(out + t * P)[p4] = a;
}

[[host_name("gemma4_ple_inject")]]
[[kernel]]
void gemma4_ple_inject(
    device const bfloat*  proj_back  [[buffer(0)]],   // (T, D)
    device const bfloat*  gamma      [[buffer(1)]],   // (D,) for this layer
    device       bfloat*  residual   [[buffer(2)]],   // (T, D) in-place
    constant uint&  T              [[buffer(3)]],
    constant uint&  D              [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    device const float* scalar_arr [[buffer(6)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  tgs [[threads_per_threadgroup]])
{
    if (row >= T) return;

    device const bfloat* xrow = proj_back + (size_t)row * D;
    device       bfloat* rrow = residual  + (size_t)row * D;

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
        rrow[i] = (bfloat)r;
    }
}

// ──────────────────────────────────────────────────────────────────────
//  gemma4_ple_gate_act_fp32_in_fp32_out
//  Reads fp32 gate (output of fp32 gemm), bf16 ple slice, writes fp32 gated.
// ──────────────────────────────────────────────────────────────────────
[[host_name("gemma4_ple_gate_act_fp32")]]
[[kernel]]
void gemma4_ple_gate_act_fp32(
    device const float*  gate            [[buffer(0)]],   // (T, P) fp32
    device const bfloat* per_layer_inputs[[buffer(1)]],   // (T, n_layers, P) bf16
    device       float*  out             [[buffer(2)]],   // (T, P) fp32
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

    float4 g  = reinterpret_cast<const device float4*>(gate + t * P)[p4];
    const uint ple_off = (t * n_layers + layer_idx) * P;
    bfloat4 ple = reinterpret_cast<const device bfloat4*>(per_layer_inputs + ple_off)[p4];

    float4 a;
    a.x = gelu_approx(g.x) * float(ple.x);
    a.y = gelu_approx(g.y) * float(ple.y);
    a.z = gelu_approx(g.z) * float(ple.z);
    a.w = gelu_approx(g.w) * float(ple.w);
    reinterpret_cast<device float4*>(out + t * P)[p4] = a;
}

// ──────────────────────────────────────────────────────────────────────
//  gemma4_ple_inject_fp32
//  Reads fp32 proj_back, bf16 gamma, bf16 residual in-place. The fp32
//  proj_back avoids the 8-bit-mantissa quantization that the bf16 version
//  imposed on the intermediate; with L1.input_layernorm.weight max=76,
//  this is the difference between rel_err 0.02 and rel_err <0.005.
// ──────────────────────────────────────────────────────────────────────
[[host_name("gemma4_ple_inject_fp32")]]
[[kernel]]
void gemma4_ple_inject_fp32(
    device const float*   proj_back  [[buffer(0)]],   // (T, D) fp32
    device const bfloat*  gamma      [[buffer(1)]],   // (D,)
    device       bfloat*  residual   [[buffer(2)]],   // (T, D) in-place
    constant uint&  T              [[buffer(3)]],
    constant uint&  D              [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    device const float* scalar_arr [[buffer(6)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  tgs [[threads_per_threadgroup]])
{
    if (row >= T) return;

    device const float*  xrow = proj_back + (size_t)row * D;
    device       bfloat* rrow = residual  + (size_t)row * D;

    threadgroup float tg_sum[32];

    float local = 0.0f;
    for (uint i = tid; i < D; i += tgs) {
        float v = xrow[i];
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
        float n = xrow[i] * invrms * (float)gamma[i];
        float r = ((float)rrow[i] + n) * scl;
        rrow[i] = (bfloat)r;
    }
}

} // namespace meow::gemma4::ple_inject
