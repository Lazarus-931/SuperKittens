// moe_group.metal — DeepSeek V2-Lite PREFILL (T>1) grouped MoE expert matvec.
//
// The default mul_mv_id path dispatches one M=1 matvec per (token, slot): every
// routed slot re-reads its expert's FULL weight slab. At prefill (T>1, ~T*top_k
// slots over 64 experts) that is ~ (T*top_k / experts_touched) redundant weight
// reads — the M4 decode kernels are bandwidth-bound, so re-reading the Q4_K /
// Q5_0 expert weights once per token dominates TTFT (moe_gate_up_q4k +
// moe_down_q5_0 = ~54% of GPU time, flat per-token = zero amortization).
//
// This file groups token-slots by expert (a single-threadgroup counting sort
// over the 64 experts) and runs a per-expert matvec that loads each weight
// K-block ONCE and dots it against every token routed to that expert (a
// register M-tile of activations). Same dot products, reordered weight loads →
// numerically equivalent; activated only for T>1 (decode T=1 keeps the default
// byte-identical path). DeepSeek-specific (own host_names, own file).
//
// host_names:
//   deepseek_moe_group_build      (counting sort: top_idx -> group_off/group_slots)
//   deepseek_mul_mv_id_q4_K_grp   (gate/up grouped matvec)
//   deepseek_mul_mv_id_q5_0_grp   (down grouped matvec)

#include "ds4_preamble.h"

#define QK_K 256
#define GRP_NEXPERT 64
#define N_R0_Q4_K 2

struct block_q2_K_grp { uchar scales[QK_K/16]; uchar qs[QK_K/4]; half d; half dmin; };
struct block_q4_K_grp { half d; half dmin; uchar scales[12]; uchar qs[QK_K/2]; };

// Counting sort over GRP_NEXPERT experts. ids = moe_top_idx [n_slots] (int32,
// expert id per slot, slot = token*top_k + k). Produces:
//   group_off    [n_expert+1]  exclusive-prefix offsets into group_slots
//   group_slots  [n_slots]     slot indices grouped by expert (stable within expert)
// Single threadgroup, 64 lanes (one per expert). n_slots = T*top_k (<= a few K).
[[host_name("deepseek_moe_group_build")]]
kernel void deepseek_moe_group_build(
        device const int32_t * ids        [[buffer(0)]],
        device       int32_t * group_off  [[buffer(1)]],
        device       int32_t * group_slots[[buffer(2)]],
        constant     uint    & n_slots    [[buffer(3)]],
        threadgroup  int     * cnt        [[threadgroup(0)]],   // [GRP_NEXPERT]
        uint tid [[thread_index_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    // 1. zero counts
    for (uint e = tid; e < GRP_NEXPERT; e += ntg) cnt[e] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. histogram (atomic over slots)
    for (uint s = tid; s < n_slots; s += ntg) {
        int e = ids[s];
        if (e >= 0 && e < GRP_NEXPERT)
            atomic_fetch_add_explicit((threadgroup atomic_int*)&cnt[e], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. exclusive prefix-sum (single lane; 64 experts is trivial)
    if (tid == 0) {
        int acc = 0;
        for (uint e = 0; e < GRP_NEXPERT; ++e) { group_off[e] = acc; acc += cnt[e]; cnt[e] = group_off[e]; }
        group_off[GRP_NEXPERT] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 4. stable scatter (single lane to preserve per-expert order = bit-identical
    //    accumulation order vs the per-slot path's independent writes).
    if (tid == 0) {
        for (uint s = 0; s < n_slots; ++s) {
            int e = ids[s];
            if (e >= 0 && e < GRP_NEXPERT) {
                int pos = cnt[e]++;
                group_slots[pos] = (int)s;
            }
        }
    }
}

// Grouped Q4_K matvec (gate/up). Grid: (ceil(out_rows / (NSG*NR0)), n_expert, 1).
// Each threadgroup owns expert e=tgpig.y and an out-row tile; it streams the
// expert's K-blocks ONCE and accumulates against every token slot routed to e.
// NR0=N_R0_Q4_K=2 rows per simdgroup. src1 = moe_x_f32 [T, d_model] (fp32).
// dst = [T, top_k, out_rows] fp32: slot s = token*top_k + k → row offset
// (s)*out_rows (== (idx + iid1*ne1)*ne0 in the per-slot kernel, ne1=top_k).
[[host_name("deepseek_mul_mv_id_q4_K_grp")]]
kernel void deepseek_mul_mv_id_q4_K_grp(
        device const char    * src0s       [[buffer(0)]],   // expert weight slabs
        device const char    * src1        [[buffer(1)]],   // moe_x_f32 [T,d_model]
        device       char    * dst         [[buffer(2)]],   // [T*top_k, out_rows]
        device const int32_t * group_off   [[buffer(3)]],
        device const int32_t * group_slots [[buffer(4)]],
        constant     uint    & in_dim      [[buffer(5)]],
        constant     uint    & out_rows    [[buffer(6)]],
        constant     uint    & top_k       [[buffer(7)]],
        constant     ulong   & nb01        [[buffer(8)]],   // weight row stride
        constant     ulong   & nb02        [[buffer(9)]],   // weight expert(slab) stride
        threadgroup  char    * shmem       [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = N_R0_Q4_K;
    const short NSG = FC_mul_mv_nsg;

    const int expert = (int)tgpig.y;
    const int beg = group_off[expert];
    const int end = group_off[expert + 1];
    if (beg >= end) return;                       // no tokens routed here

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = (int)in_dim / QK_K;
    const int first_row = ((int)tgpig.x * NSG + sgitg) * NR0;

    // Weight base for this expert's row tile.
    device const block_q4_K_grp * x0 =
        (device const block_q4_K_grp *)(src0s + (ulong)expert*nb02 + (ulong)first_row*nb01);

    const uint in_floats = in_dim;               // src1 row stride (fp32 contiguous)

    // M-tile over the expert's tokens, MT slots at a time, so each loaded K-block
    // serves MT activation rows. MT chosen to fit registers (yl/yh per slot).
    constexpr int MT = 4;
    for (int mb = beg; mb < end; mb += MT) {
        const int mcnt = min(MT, end - mb);

        // resolve token rows for this M-tile
        device const float * yptr[MT];
        device float       * dptr[MT];
        for (int j = 0; j < MT; ++j) {
            const int slot = (j < mcnt) ? group_slots[mb + j] : group_slots[mb];  // pad reuses slot0
            const int token = slot / (int)top_k;
            yptr[j] = (device const float *)(src1) + (ulong)token*in_floats;
            dptr[j] = (device float *)(dst) + (ulong)slot*out_rows;
        }

        float sumf[MT][NR0];
        for (int j = 0; j < MT; ++j)
            for (short r = 0; r < NR0; ++r) sumf[j][r] = 0.f;

        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

        for (int ib = ix; ib < nb; ib += 4) {
            // Per-slot activation slices for this K-block.
            float yl[MT][16];
            float yh[MT][16];
            float sumy[MT][4];
            for (int j = 0; j < mcnt; ++j) {
                device const float * y4 = yptr[j] + ib*QK_K + 64*iq + 8*ir;
                sumy[j][0]=sumy[j][1]=sumy[j][2]=sumy[j][3]=0.f;
                for (short i = 0; i < 8; ++i) {
                    yl[j][i+0]=y4[i+  0]; sumy[j][0]+=yl[j][i+0];
                    yl[j][i+8]=y4[i+ 32]; sumy[j][1]+=yl[j][i+8];
                    yh[j][i+0]=y4[i+128]; sumy[j][2]+=yh[j][i+0];
                    yh[j][i+8]=y4[i+160]; sumy[j][3]+=yh[j][i+8];
                }
            }
            for (short row = 0; row < NR0; ++row) {
                device const block_q4_K_grp * xr =
                    (device const block_q4_K_grp *)((device const char *)x0 + (ulong)row*nb01);
                device const block_q4_K_grp * xb = xr + ib;
                device const uint16_t * sc = (device const uint16_t *)xb->scales + iq;
                device const uint16_t * q1 = (device const uint16_t *)xb->qs + 16*iq + 4*ir;
                device const half * dh = &xb->d;
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const uint16_t * q2 = q1 + 32;
                const float dall = dh[0];
                const float dmin = dh[1];
                for (int j = 0; j < mcnt; ++j) {
                    float4 acc1 = {0.f,0.f,0.f,0.f};
                    float4 acc2 = {0.f,0.f,0.f,0.f};
                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[j][2*i+0] * (q1[i] & 0x000F);
                        acc1[1] += yl[j][2*i+1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[j][2*i+8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[j][2*i+9] * (q1[i] & 0xF000);
                        acc2[0] += yh[j][2*i+0] * (q2[i] & 0x000F);
                        acc2[1] += yh[j][2*i+1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[j][2*i+8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[j][2*i+9] * (q2[i] & 0xF000);
                    }
                    sumf[j][row] += dall * ((acc1[0] + 1.f/256.f*acc1[1]) * sc8[0] +
                                            (acc1[2] + 1.f/256.f*acc1[3]) * sc8[1] * 1.f/16.f +
                                            (acc2[0] + 1.f/256.f*acc2[1]) * sc8[4] +
                                            (acc2[2] + 1.f/256.f*acc2[3]) * sc8[5] * 1.f/16.f) -
                                    dmin * (sumy[j][0]*sc8[2] + sumy[j][1]*sc8[3] +
                                            sumy[j][2]*sc8[6] + sumy[j][3]*sc8[7]);
                }
            }
        }

        // reduce + write each (slot, row)
        for (int j = 0; j < mcnt; ++j)
            for (short row = 0; row < NR0 && first_row + row < (int)out_rows; ++row) {
                float tot = simd_sum(sumf[j][row]);
                if (tiisg == 0) dptr[j][first_row + row] = tot;
            }
    }
    (void)shmem;
}

// Grouped Q5_0 matvec (down). Grid: (ceil(out_rows/(NSG*NR0)), n_expert, 1).
// Mirrors q5_0_mv_impl's per-lane fixed-8-slice layout but reuses each weight
// block across the expert's M-tile of tokens. NR0=N_R0_Q5_0=4.
[[host_name("deepseek_mul_mv_id_q5_0_grp")]]
kernel void deepseek_mul_mv_id_q5_0_grp(
        device const char    * src0s       [[buffer(0)]],
        device const char    * src1        [[buffer(1)]],   // moe_mid_f32 [T,top_k,in_dim]
        device       char    * dst         [[buffer(2)]],   // [T*top_k, out_rows]
        device const int32_t * group_off   [[buffer(3)]],
        device const int32_t * group_slots [[buffer(4)]],
        constant     uint    & in_dim      [[buffer(5)]],
        constant     uint    & out_rows    [[buffer(6)]],
        constant     uint    & top_k       [[buffer(7)]],
        constant     ulong   & nb01        [[buffer(8)]],
        constant     ulong   & nb02        [[buffer(9)]],
        threadgroup  char    * shmem       [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = N_R0_Q5_0;
    constexpr short NQ = 8;
    const short NSG = FC_mul_mv_nsg;

    const int expert = (int)tgpig.y;
    const int beg = group_off[expert];
    const int end = group_off[expert + 1];
    if (beg >= end) return;

    const int nb = (int)in_dim / QK5_0;
    const int first_row = ((int)tgpig.x * NSG + sgitg) * NR0;

    const short ix    = tiisg/(N_SIMDWIDTH/NQ);   // 0..7
    const short il    = tiisg%(N_SIMDWIDTH/NQ);   // 0..3
    const short base  = il*NQ;
    const short shift = (base < 16) ? 0 : 4;
    const short boff  = base & 15;

    device const block_q5_0 * x0 =
        (device const block_q5_0 *)(src0s + (ulong)expert*nb02 + (ulong)first_row*nb01);

    // src1 here is moe_mid_f32 laid out [T, top_k, in_dim] = [slot, in_dim],
    // i.e. row index == slot directly (the swiglu writes per slot).
    const uint in_floats = in_dim;

    constexpr int MT = 4;
    for (int mb = beg; mb < end; mb += MT) {
        const int mcnt = min(MT, end - mb);
        device const float * yptr[MT];
        device float       * dptr[MT];
        for (int j = 0; j < MT; ++j) {
            const int slot = (j < mcnt) ? group_slots[mb + j] : group_slots[mb];
            yptr[j] = (device const float *)(src1) + (ulong)slot*in_floats;
            dptr[j] = (device float *)(dst) + (ulong)slot*out_rows;
        }

        float sumf[MT][NR0];
        for (int j = 0; j < MT; ++j)
            for (short r = 0; r < NR0; ++r) sumf[j][r] = 0.f;

        for (int ib = ix; ib < nb; ib += NQ) {
            float yl[MT][NQ];
            for (int j = 0; j < mcnt; ++j) {
                device const float * yb = yptr[j] + ib*QK5_0 + base;
                for (short i = 0; i < NQ; ++i) yl[j][i] = yb[i];
            }
            for (short row = 0; row < NR0; ++row) {
                device const block_q5_0 * xr =
                    (device const block_q5_0 *)((device const char *)x0 + (ulong)row*nb01);
                device const block_q5_0 * xb = xr + ib;
                device const uint8_t * qs = xb->qs + boff;
                uint32_t qh = (uint32_t)xb->qh[0] | ((uint32_t)xb->qh[1]<<8)
                            | ((uint32_t)xb->qh[2]<<16) | ((uint32_t)xb->qh[3]<<24);
                qh >>= base;
                const float d = float(xb->d);
                for (int j = 0; j < mcnt; ++j) {
                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NQ; ++i) {
                        const int lo = (qs[i] >> shift) & 0x0F;
                        const int hi = (int)((qh >> i) & 1) << 4;
                        sumq += float((lo | hi) - 16) * yl[j][i];
                    }
                    sumf[j][row] += sumq * d;
                }
            }
        }

        for (int j = 0; j < mcnt; ++j)
            for (short row = 0; row < NR0 && first_row + row < (int)out_rows; ++row) {
                float tot = simd_sum(sumf[j][row]);
                if (tiisg == 0) dptr[j][first_row + row] = tot;
            }
    }
    (void)shmem;
}
