// mul_mv_id.metal — DeepSeek V2-Lite MoE per-expert matvec dispatch.
//
// Ported from ds4 /metal/moe.metal line ~861 (kernel_mul_mv_id<...q8_0...>).
// This iteration ports only the Q8_0 variant — V2-Lite shared experts and
// initial routed-expert pipeline are Q8_0; q2_K / q4_K / iq2_xxs variants will
// land in a follow-up alongside their numpy refs.
//
// SK host_name: "deepseek_mul_mv_id_q8_0".

#include "ds4_preamble.h"

struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
};

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        thread float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

// Per-row Q8_0 matvec inner loop — same reduction shape as the dense ds4 path.
template<short NR0>
static inline void q8_0_mv_impl(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00 / QK8_0;

    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const float * y = (device const float *)(src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 +
                                 (i12/args.r2)*args.nb02 +
                                 (i13/args.r3)*args.nb03;
        ax[row] = (device const block_q8_0 *)((device char *)src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int   ib0 = sgitg*NQ + ix;

    float yl[NQ];
    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];

        for (short row = 0; row < NR0; ++row) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }
            sumf[row] += sumq * float(ax[row][ib].d);
        }
        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *)dst +
        (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Decode-time expert matvec.  The ids tensor selects the routed expert for each
// slot (tgpig.z), then the q8_0 row kernel runs against that expert's weight
// slice without CPU-side per-expert dispatches.
[[host_name("deepseek_mul_mv_id_q8_0")]]
kernel void deepseek_mul_mv_id_q8_0(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;
    device char *       dst_cur  = dst   + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    q8_0_mv_impl<N_R0_Q8_0>(args0, src0_cur, src1_cur, dst_cur, shmem, tgpig, tiisg, sgitg);
}
