//
//  pre_ssm.metal
//  SuperKittens — Mamba-3 pre-SSM: norm(Q/K/B) + rotary(Q/K)
//
//  SIMD-group per position. Fused norm + rotary.

#include <metal_stdlib>
using namespace metal;

enum : uint { THREADS = 128, ROWS = 4 };

[[host_name("mamba3_pre_ssm")]]
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void mamba3_pre_ssm(
    device const half* xBC,       // (BH, L, 2*DQ): Q/K then B
    device const half* dt,        // (BH, L) — cumsum A for rotary
    device const half* angle,     // (BH, L, DQ/2)
    device const half* norm_w,    // (DQ,)
    device half* Q_out, device half* K_out, device half* V_out,
    device half* A_out, device half* B_out,
    constant uint& BH, constant uint& L, constant uint& DQ, constant float& eps,
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x, pos = gid.y * ROWS + simd;
    if (pos >= L) return;

    const uint HD = DQ / 2;
    const float PI = 3.141592653589793f;
    const size_t o2 = ((size_t)bh * L + pos) * (DQ * 2);
    const size_t o1 = ((size_t)bh * L + pos) * DQ;
    const size_t os = (size_t)bh * L + pos;
    const size_t oa = ((size_t)bh * L + pos) * HD;

    // ── RMSNorm on Q/K (first DQ) ──
    float sq = 0;
    for (uint i = lane; i < DQ; i += 32) { float v = float(xBC[o2 + i]); sq += v * v; }
    sq = simd_sum(sq);
    float inv_rms = metal::precise::rsqrt(sq / float(DQ) + eps);

    // ── RMSNorm on B (second DQ) ──
    float sq_b = 0;
    for (uint i = lane; i < DQ; i += 32) { float v = float(xBC[o2 + DQ + i]); sq_b += v * v; }
    sq_b = simd_sum(sq_b);
    float inv_b = metal::precise::rsqrt(sq_b / float(DQ) + eps);

    float dt_val = float(dt[os]);

    // ── Apply norm + rotary to Q/K, write V ──
    for (uint i = lane; i < HD; i += 32) {
        float qi = float(xBC[o2 + i]) * inv_rms * float(norm_w[i]);
        float ki = float(xBC[o2 + i + HD]) * inv_rms * float(norm_w[i + HD]);

        float th = dt_val * float(angle[oa + i]) * PI;
        float cs = metal::fast::cos(th), sn = metal::fast::sin(th);

        float q_rot = qi * cs - ki * sn;
        float k_rot = qi * sn + ki * cs;

        Q_out[o1 + i]      = half(q_rot);
        Q_out[o1 + i + HD] = half(k_rot);
        K_out[o1 + i]      = half(q_rot);
        K_out[o1 + i + HD] = half(k_rot);
        V_out[o1 + i]      = half(q_rot);
        V_out[o1 + i + HD] = half(k_rot);
    }

    // ── Write A, B ──
    if (lane == 0) {
        A_out[os] = half(dt_val);
        B_out[os] = half(float(xBC[o2 + DQ]) * inv_b * float(norm_w[0]));
    }
    for (uint i = lane; i < DQ; i += 32) {
        float bv = float(xBC[o2 + DQ + i]) * inv_b * float(norm_w[i]);
        B_out[o1 + i] = half(bv);  // B is (BH, L, DQ) — write full vector
    }
}
