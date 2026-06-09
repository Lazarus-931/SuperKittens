// gemm_mma_smallm.metal — small-M (seq 2..8) multi-RHS matvec.
//
// gemm_mma stages a [BK][BN] weight tile in threadgroup memory and runs an
// 8-row MMA per K-step; at M<8 the dequant + transpose + barriers cost is
// M-independent (a fixed floor ~5-6x a seq=1 matvec), so a seq=K verify forward
// is ~5x a decode instead of ~1x. This kernel reuses the seq=1 matvec geometry
// (no transpose, no TG weight staging, fp32 accumulate, 128 threads) and reads
// each weight block ONCE while multiplying it into all M activation rows held in
// registers — so weight bandwidth amortizes M-fold like the MMA, but without the
// MMA fixed floor. Output stays bandwidth-bound for M<=MAXM.
//
// Bindings (mirror gemm_mma): 0:A fp16 [M,K]  1:W quant [N,K]  2:C fp16 [M,N]
//   3:M 4:N 5:K 6:ldC.  Dispatch: grid (ceil(N/NR0),1,1), tg (NW*NSG,1,1).

#include <metal_stdlib>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

namespace sksm {

constant constexpr short NW   = 32;
constant constexpr short NSG  = 4;     // Q4_K sm: 4 simdgroups (row/sub-block split)
// Q8_0 sm: one simdgroup per TG. At M=2..8 on these small projection shapes the
// kernel is occupancy-bound, not weight-bandwidth-bound (~27% of roofline at
// M=8 with NSG=4): NSG=4 packs 128 threads into ceil(N/NR0) TGs, which starves
// the M4 scheduler. NSG=1 launches the same TG count at 32 threads each → finer
// work granularity, +20-36% across qkv/gate/up and the (tail-dominating) LM head.
constant constexpr short NSG_Q8 = 1;
constant constexpr short NR0  = 2;
constant constexpr short NQ   = 8;
constant constexpr uint  MAXM = 8;   // register accumulator cap

struct __attribute__((packed)) q8_block { half d; int8_t qs[32]; };

struct block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
};

} // namespace sksm
using namespace sksm;

// ── Q8_0 multi-RHS. Lane partition matches q8_0_matvec (NQ groups of NW/NQ
// lanes); each lane accumulates M dot-products against the SAME weight slice.
kernel void gemm_mma_q8_0_sm(
    device const half*  A    [[buffer(0)]],   // [M,K] row-major, ldA = K
    device const uchar* W    [[buffer(1)]],   // [N,K] q8_0
    device       half*  C    [[buffer(2)]],   // [M,N] row stride ldC
    constant uint&      M    [[buffer(3)]],
    constant uint&      N    [[buffer(4)]],
    constant uint&      K    [[buffer(5)]],
    constant uint&      ldC  [[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb   = K / 32;
    const uint row0 = tgpig.x * NR0;

    const ushort ix  = tiisg / (NW / NQ);     // 0..7 (which 8-wide block)
    const ushort il  = tiisg % (NW / NQ);     // 0..3 (which 8 elems inside)
    const uint   ib0 = (uint)sgitg * NQ + ix;
    const uint   koff = il * NQ;              // element offset of this lane's 8

    device const q8_block* ax[NR0];
    #pragma unroll
    for (short r = 0; r < NR0; ++r) {
        const uint row = row0 + r;
        ax[r] = (device const q8_block*)(W + (size_t)row * nb * sizeof(q8_block));
    }

    float sumf[NR0][MAXM];
    #pragma unroll
    for (short r = 0; r < NR0; ++r)
        for (uint m = 0; m < MAXM; ++m) sumf[r][m] = 0.f;

    for (uint ib = ib0; ib < nb; ib += NSG_Q8 * NQ) {
        const uint kbase = ib * 32 + koff;    // global K offset of this lane's 8
        // Cast both rows' 8 int8 weights to float ONCE (out of the m-loop);
        // the per-row inner loop is then NQ fma against shared float weights,
        // matching the matvec's (sum int8*y)*d order so the result stays exact.
        float wf[NR0][NQ];
        float dr[NR0];
        #pragma unroll
        for (short r = 0; r < NR0; ++r) {
            if (row0 + (uint)r >= N) { dr[r] = 0.f; continue; }
            device const int8_t* qs = ax[r][ib].qs + koff;
            #pragma unroll
            for (short i = 0; i < NQ; ++i) wf[r][i] = (float)qs[i];
            dr[r] = (float)ax[r][ib].d;
        }
        for (uint m = 0; m < M; ++m) {
            device const half* yb = A + (size_t)m * K + kbase;
            float ylm[NQ];
            #pragma unroll
            for (short i = 0; i < NQ; ++i) ylm[i] = (float)yb[i];
            #pragma unroll
            for (short r = 0; r < NR0; ++r) {
                float sq = 0.f;
                #pragma unroll
                for (short i = 0; i < NQ; ++i) sq += wf[r][i] * ylm[i];
                sumf[r][m] += sq * dr[r];
            }
        }
    }

    // Reduce across lanes (simdgroup) then across the NSG_Q8 simdgroups per row.
    threadgroup float shmem[NR0 * NSG_Q8 * MAXM];
    for (uint m = 0; m < M; ++m) {
        #pragma unroll
        for (short r = 0; r < NR0; ++r) {
            float v = simd_sum(sumf[r][m]);
            if (tiisg == 0) shmem[(r * NSG_Q8 + sgitg) * MAXM + m] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (uint m = 0; m < M; ++m) {
            #pragma unroll
            for (short r = 0; r < NR0; ++r) {
                float v = (tiisg < NSG_Q8) ? shmem[(r * NSG_Q8 + tiisg) * MAXM + m] : 0.0f;
                v = simd_sum(v);
                if (tiisg == 0) {
                    const uint out_row = row0 + r;
                    if (out_row < N) C[(size_t)m * ldC + out_row] = (half)v;
                }
            }
        }
    }
}

// ── Q4_K multi-RHS. Reuses the GGML tap layout from q4k_matvec but holds M
// activation register sets and accumulates M sums per lane against one weight
// decode. Each lane covers super-blocks at stride NSG-pair like the matvec.
kernel void gemm_mma_q4k_sm(
    device const half*  A    [[buffer(0)]],
    device const uchar* W    [[buffer(1)]],
    device       half*  C    [[buffer(2)]],
    constant uint&      M    [[buffer(3)]],
    constant uint&      N    [[buffer(4)]],
    constant uint&      K    [[buffer(5)]],
    constant uint&      ldC  [[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint nb   = K / QK_K;
    const uint row0 = tgpig.x * NR0;

    const short row_off   = sgitg / 2;
    const short sg_in_row = sgitg % 2;
    const uint  row = row0 + row_off;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    device const block_q4_K* xrow = (row < N)
        ? (device const block_q4_K*)(W + (size_t)row * nb * sizeof(block_q4_K))
        : (device const block_q4_K*)W;

    const uint ib_start  = (uint)ix + 4u * (uint)sg_in_row;
    const uint ib_stride = 4u * 2u;

    float sumf[MAXM];
    #pragma unroll
    for (uint m = 0; m < MAXM; ++m) sumf[m] = 0.f;

    uint16_t sc16[4];
    thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

    for (uint ib = ib_start; row < N && ib < nb; ib += ib_stride) {
        // Weight decode for this super-block — done ONCE, reused over all M.
        device const uint16_t* sc = (device const uint16_t*)xrow[ib].scales + iq;
        device const uint16_t* q1 = (device const uint16_t*)xrow[ib].qs + 16 * iq + 4 * ir;
        device const uint16_t* q2 = q1 + 32;
        device const half*     dh = &xrow[ib].d;
        sc16[0] = sc[0] & kmask1;
        sc16[1] = sc[2] & kmask1;
        sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
        sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
        const float d    = (float)dh[0];
        const float dmin = (float)dh[1];

        for (uint m = 0; m < M; ++m) {
            device const half* y4_h = A + (size_t)m * K + ib * QK_K + 64 * iq + 8 * ir;
            float yl[16], yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = (float)y4_h[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = (float)y4_h[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = (float)y4_h[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = (float)y4_h[i + 160]; sumy[3] += yh[i + 8];
            }
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[m] += d * ((acc1[0] + (1.f/256.f) * acc1[1]) * sc8[0] +
                            (acc1[2] + (1.f/256.f) * acc1[3]) * sc8[1] * (1.f/16.f) +
                            (acc2[0] + (1.f/256.f) * acc2[1]) * sc8[4] +
                            (acc2[2] + (1.f/256.f) * acc2[3]) * sc8[5] * (1.f/16.f))
                     - dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                               sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
    }

    // Reduce within SG, then across the SG pair sharing the row.
    threadgroup float shmem[NR0 * 2 * MAXM];   // [row_off][sg_in_row][m]
    for (uint m = 0; m < M; ++m) {
        const float sg_sum = simd_sum(sumf[m]);
        if (tiisg == 0) shmem[(row_off * 2 + sg_in_row) * MAXM + m] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg < NR0 && tiisg == 0) {
        const uint out_row = row0 + sgitg;
        if (out_row < N) {
            for (uint m = 0; m < M; ++m) {
                C[(size_t)m * ldC + out_row] =
                    (half)(shmem[(sgitg * 2 + 0) * MAXM + m] +
                           shmem[(sgitg * 2 + 1) * MAXM + m]);
            }
        }
    }
}
