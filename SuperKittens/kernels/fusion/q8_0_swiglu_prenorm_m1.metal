// q8_0_swiglu_prenorm_m1.metal — Fused (residual + RMSNorm + gate+up Q8_0 matvec + SiLU·mul) for M=1 decode.
//
// Replaces add_rmsnorm → q8_0_swiglu_m1 (2 dispatches + 1 fence) with a single
// dispatch. Each TG covers SW_NR0=2 output rows and ALL of K=d_model in 128
// threads. Sum-of-squares of (x+delta) is accumulated as a side product of the
// matvec inner loop (each thread reads K/128 unique x elts), then reduced
// inside the TG to inv_rms. The matvec accumulators are multiplied by
// (gamma_i * inv_rms) via the identity:
//     dot(W_row, (x+δ) ⊙ γ * inv_rms) = inv_rms * Σ q_i · d_i · γ_i · (x_i+δ_i)
// Only TG with tgpig.x == 0 writes y_attn[i] = (x+δ)[i] for the layer's
// final residual; other TGs skip the write (each x index would otherwise be
// stored ~n_int/2 times redundantly).

#include <metal_stdlib>
using namespace metal;

#ifndef Q8_BLOCK
#define Q8_BLOCK 32
#endif

struct __attribute__((packed)) q8_block_pn { half d; int8_t qs[Q8_BLOCK]; };

constant constexpr short PN_NW  = 32;
constant constexpr short PN_NSG = 4;
constant constexpr short PN_NR0 = 2;
constant constexpr short PN_NQ  = 8;

[[host_name("q8_0_swiglu_prenorm_m1")]]
kernel void q8_0_swiglu_prenorm_m1(
    device const half      *X       [[buffer(0)]],   // x, fp16 [K]
    device const half      *D       [[buffer(1)]],   // delta (o_proj), fp16 [K]
    device const half      *G       [[buffer(2)]],   // gamma (pre-MLP norm γ), fp16 [K]
    device const uchar     *Ag_raw  [[buffer(3)]],   // W_gate, Q8_0 [N, K]
    device const uchar     *Au_raw  [[buffer(4)]],   // W_up,   Q8_0 [N, K]
    device       half      *Y       [[buffer(5)]],   // y_attn out, fp16 [K]
    device       half      *C       [[buffer(6)]],   // up_buf (swiglu out), fp16 [N]
    constant uint           &K      [[buffer(7)]],
    constant uint           &N      [[buffer(8)]],
    constant float          &eps    [[buffer(9)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb   = K / Q8_BLOCK;
    const uint row0 = tgpig.x * PN_NR0;
    const bool write_y = (tgpig.x == 0u);

    const ushort ix = tiisg / (PN_NW / PN_NQ);
    const ushort il = tiisg % (PN_NW / PN_NQ);
    const uint   ib0 = (uint)sgitg * PN_NQ + ix;

    device const q8_block_pn *axg[PN_NR0];
    device const q8_block_pn *axu[PN_NR0];
    #pragma unroll
    for (short r = 0; r < PN_NR0; ++r) {
        const uint row = row0 + r;
        axg[r] = (device const q8_block_pn *)(Ag_raw + (size_t)row * nb * sizeof(q8_block_pn));
        axu[r] = (device const q8_block_pn *)(Au_raw + (size_t)row * nb * sizeof(q8_block_pn));
    }

    float sumg[PN_NR0] = {0.f};
    float sumu[PN_NR0] = {0.f};
    float sumSq = 0.f;
    half  yl[PN_NQ];
    half  gl[PN_NQ];

    for (uint ib = ib0; ib < nb; ib += PN_NSG * PN_NQ) {
        const uint base = ib * Q8_BLOCK + il * PN_NQ;
        device const half *xb = X + base;
        device const half *db = D + base;
        device const half *gb = G + base;

        // Compute (x+δ), accumulate sum_sq, scale by γ; store γ·(x+δ) in yl.
        #pragma unroll
        for (short i = 0; i < PN_NQ; ++i) {
            float xv = (float)xb[i] + (float)db[i];
            sumSq += xv * xv;
            yl[i] = (half)xv;
            gl[i] = gb[i];
        }

        // Write y_attn from TG 0 only.
        if (write_y) {
            device half *yo = Y + base;
            #pragma unroll
            for (short i = 0; i < PN_NQ; ++i) yo[i] = yl[i];
        }

        #pragma unroll
        for (short r = 0; r < PN_NR0; ++r) {
            device const int8_t *qsg = axg[r][ib].qs + il * PN_NQ;
            device const int8_t *qsu = axu[r][ib].qs + il * PN_NQ;
            float sg = 0.f, su = 0.f;
            #pragma unroll
            for (short i = 0; i < PN_NQ; ++i) {
                float xv = (float)yl[i] * (float)gl[i];
                sg += (float)qsg[i] * xv;
                su += (float)qsu[i] * xv;
            }
            sumg[r] += sg * (float)axg[r][ib].d;
            sumu[r] += su * (float)axu[r][ib].d;
        }
    }

    threadgroup float shg[PN_NR0 * PN_NSG];
    threadgroup float shu[PN_NR0 * PN_NSG];
    threadgroup float ssh[PN_NSG];

    // SimD-level reductions.
    float ss = simd_sum(sumSq);
    #pragma unroll
    for (short r = 0; r < PN_NR0; ++r) {
        float vg = simd_sum(sumg[r]);
        float vu = simd_sum(sumu[r]);
        if (tiisg == 0) {
            shg[r * PN_NSG + sgitg] = vg;
            shu[r * PN_NSG + sgitg] = vu;
        }
    }
    if (tiisg == 0) ssh[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG reduction inside SG 0.
    float total_sq;
    if (sgitg == 0) {
        float v = (tiisg < PN_NSG) ? ssh[tiisg] : 0.f;
        v = simd_sum(v);
        if (tiisg == 0) ssh[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_sq = ssh[0];
    float inv_rms = metal::precise::rsqrt(total_sq / float(K) + eps);

    if (sgitg == 0) {
        #pragma unroll
        for (short r = 0; r < PN_NR0; ++r) {
            float vg = (tiisg < PN_NSG) ? shg[r * PN_NSG + tiisg] : 0.0f;
            float vu = (tiisg < PN_NSG) ? shu[r * PN_NSG + tiisg] : 0.0f;
            vg = simd_sum(vg);
            vu = simd_sum(vu);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) {
                    float g = vg * inv_rms;
                    float u = vu * inv_rms;
                    float sig = 1.0f / (1.0f + exp(-g));
                    C[out_row] = (half)(g * sig * u);
                }
            }
        }
    }
}
