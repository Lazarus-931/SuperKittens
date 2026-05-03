//
//  conv2d.metal — TM=64 (8×8 spatial), TN=64, BK=64.  49 threadgroups.
//  64 threads, 2 SIMD groups, MR=4.  24KB threadgroup.
//  Fewer threadgroups = fewer total barriers = faster.
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct ConvParams { uint N, H, W, C_in, K_H, K_W, C_out, H_out, W_out; };
enum : uint { TM = 32, TN = 64, BK = 64, MR = 2, MC = 8, HT = 4, WT = 8 };

[[host_name("conv2d_tiled")]]
[[kernel]]
void conv2d_tiled(
    device const half* x       [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device const half* bias    [[buffer(2)]],
    device half* y             [[buffer(3)]],
    constant ConvParams& p     [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint n  = gid.z;
    const uint h0 = gid.y * HT;
    const uint w0 = gid.x * WT;
    if (n >= p.N) return;

    threadgroup half As[TM * BK];   // 64×64 = 8KB
    threadgroup half Bs[BK * TN];   // 64×64 = 8KB
    threadgroup half Cs[TM * TN];   // 64×64 = 8KB → 24KB

    const uint n_a = TM * BK, n_b = BK * TN;
    const uint tid = simd * 32 + lane;

    const device half4* x4 = reinterpret_cast<const device half4*>(x);
    const device half4* w4 = reinterpret_cast<const device half4*>(weight);

    simdgroup_float8x8 acc[MR][MC] = {};
    const uint r0 = simd * 16;  // each SIMD handles 16 rows

    for (uint kh = 0; kh < p.K_H; kh++) {
        for (uint kw = 0; kw < p.K_W; kw++) {

            for (uint i = tid; i < n_a / 4; i += 64) {
                uint r = (i * 4) / BK;
                uint c = (i * 4) % BK;
                uint r_h = r / WT, r_w = r % WT;
                uint hi = h0 + r_h + kh, wi = w0 + r_w + kw;
                reinterpret_cast<threadgroup half4*>(As)[i] =
                    (hi < p.H && wi < p.W)
                        ? x4[(((n * p.H + hi) * p.W + wi) * p.C_in + c) / 4]
                        : half4(0);
            }

            for (uint i = tid; i < n_b / 4; i += 64) {
                uint r = (i * 4) / TN;
                uint c = (i * 4) % TN;
                reinterpret_cast<threadgroup half4*>(Bs)[i] =
                    (r < p.C_in && c < p.C_out)
                        ? w4[(((kh * p.K_W + kw) * p.C_in + r) * p.C_out + c) / 4]
                        : half4(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k = 0; k < BK / 8; k++) {
                for (uint r = 0; r < MR; r++) {
                    simdgroup_half8x8 a;
                    simdgroup_load(a, As + (r0 + r * 8) * BK + k * 8, BK);
                    for (uint c = 0; c < MC; c++) {
                        simdgroup_half8x8 b;
                        simdgroup_load(b, Bs + k * 8 * TN + c * 8, TN);
                        simdgroup_multiply_accumulate(acc[r][c], a, b, acc[r][c]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store → Cs
    for (uint r = 0; r < MR; r++) {
        for (uint c = 0; c < MC; c++) {
            float2 v = reinterpret_cast<thread float2&>(acc[r][c].thread_elements());
            uint qid = lane / 4;
            uint lr = (qid & 4) + ((lane / 2) % 4);
            uint lc = (qid & 2) * 2 + (lane % 2) * 2;
            Cs[(r0 + r * 8 + lr) * TN + c * 8 + lc]     = half(v.x);
            Cs[(r0 + r * 8 + lr) * TN + c * 8 + lc + 1] = half(v.y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output with bias
    device half4* y4 = reinterpret_cast<device half4*>(y);
    const device half4* b4 = reinterpret_cast<const device half4*>(bias);

    for (uint i = tid; i < (TM * TN) / 4; i += 64) {
        uint r = (i * 4) / TN;
        uint c = (i * 4) % TN;
        uint r_h = r / WT, r_w = r % WT;
        uint h_out = h0 + r_h, w_out = w0 + r_w;
        if (h_out >= p.H_out || w_out >= p.W_out || c >= p.C_out) continue;

        half4 val = reinterpret_cast<threadgroup half4*>(Cs)[i];
        if (bias) val = half4(float4(val) + float4(b4[c / 4]));
        y4[(((n * p.H_out + h_out) * p.W_out + w_out) * p.C_out + c) / 4] = val;
    }
}
