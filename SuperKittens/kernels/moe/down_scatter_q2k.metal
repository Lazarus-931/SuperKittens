// moe_down_scatter_q2k.metal — Q2_K-packed version of moe_down_scatter.

#include <metal_stdlib>
using namespace metal;

#define QK_K 256

struct block_q2_K {
    uchar scales[QK_K/16]; // 16
    uchar qs[QK_K/4];      // 64
    half  d;
    half  dmin;
};

constant constexpr uint COLS_PER_SG = 2;
constant constexpr uint N_SG        = 8;
constant constexpr uint COLS_PER_TG = COLS_PER_SG * N_SG;   // 16
constant constexpr uint TGSZ        = 32 * N_SG;            // 256
constant constexpr uint MAX_TOPK    = 16;

[[host_name("moe_down_scatter_q2k")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void moe_down_scatter_q2k(
    device const half*  hidden    [[buffer(0)]],
    device const char*  W_down    [[buffer(1)]],   // Q2_K bytes
    device const int*   exp_ids   [[buffer(2)]],
    device const half*  route_w   [[buffer(3)]],
    device const half*  residual  [[buffer(4)]],
    device half*        out       [[buffer(5)]],
    constant uint&      T         [[buffer(6)]],
    constant uint&      top_k     [[buffer(7)]],
    constant uint&      D         [[buffer(8)]],
    constant uint&      N_int     [[buffer(9)]],
    uint3   gid                   [[threadgroup_position_in_grid]],
    uint    tid                   [[thread_index_in_threadgroup]],
    ushort  sg                    [[simdgroup_index_in_threadgroup]],
    ushort  lane                  [[thread_index_in_simdgroup]])
{
    const uint t        = gid.y;
    const uint col_base = gid.x * COLS_PER_TG;
    if (t >= T) return;

    // n_int MUST be a multiple of QK_K (256) — Q2_K's atomic quant block size.
    // If a future model exposes a non-multiple n_int, this int-divide silently
    // truncates the tail (same class of bug as the fp16 down_scatter prior to
    // its tile round-up fix). All current SK targets (DSv2-Lite n_int=1408,
    // Qwen3-MoE-A3B n_int=768) divide cleanly. Keeping as a divide for now;
    // if you add a model where N_int % 256 != 0 you must rework the tail.
    const uint nb = N_int / QK_K;
    const ulong row_bytes = (ulong)nb * sizeof(block_q2_K);

    // Cache exp_ids / route_w in tgmem.
    threadgroup int   exp_tg[MAX_TOPK];
    threadgroup half  rw_tg[MAX_TOPK];
    if (tid < top_k) {
        exp_tg[tid] = exp_ids[t * top_k + tid];
        rw_tg[tid]  = route_w[t * top_k + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // q2k lane layout (matches q2k_matvec.metal / ds4).
    const short ix = lane / 8;       // 0..3 — block-step within stride-4
    const short it = lane % 8;       // 0..7
    const short iq = it / 4;         // 0 or 1 — half of QK_K
    const short ir = it % 4;         // 0..3
    const short is = (8 * ir) / 16;  // 0 or 1

    float acc_col[COLS_PER_SG] = {0.f, 0.f};

    // Loop over the COLS_PER_SG output dims this simdgroup handles.
    for (uint cs = 0; cs < COLS_PER_SG; cs++) {
        const uint d_out = col_base + sg * COLS_PER_SG + cs;
        if (d_out >= D) continue;

        float sum_slot = 0.f;

        for (uint s = 0; s < top_k; s++) {
            const int   eid = exp_tg[s];
            const float rw  = float(rw_tg[s]);

            // hidden for this (t, s) slot.
            device const half* y_base =
                hidden + (t * top_k + s) * N_int + ix * QK_K + 128 * iq + 8 * ir;

            // W_down for this (eid, d_out) row.
            device const block_q2_K* xb = (device const block_q2_K*)(
                W_down + ((ulong)eid * D + d_out) * row_bytes);

            float sumf = 0.f;
            float yl[32];

            for (uint ib = ix; ib < nb; ib += 4) {
                float4 sumy = {0.f, 0.f, 0.f, 0.f};
                device const half* y4 = y_base + (ib - ix) * QK_K;
                // Actually, the q2k_matvec advances y4 by 4*QK_K per inner step,
                // starting at y_base. For a generic ib we need y_base + ib*QK_K
                // minus ix*QK_K (since y_base already includes ix*QK_K).
                // Simpler: recompute.
                y4 = hidden + (t * top_k + s) * N_int + ib * QK_K + 128 * iq + 8 * ir;

                for (short i = 0; i < 8; ++i) {
                    yl[i +  0] = (float)y4[i +  0]; sumy[0] += yl[i +  0];
                    yl[i +  8] = (float)y4[i + 32]; sumy[1] += yl[i +  8];
                    yl[i + 16] = (float)y4[i + 64]; sumy[2] += yl[i + 16];
                    yl[i + 24] = (float)y4[i + 96]; sumy[3] += yl[i + 24];
                }

                device const uint8_t  * sc = (device const uint8_t  *)xb[ib].scales + 8 * iq + is;
                device const uint16_t * qs = (device const uint16_t *)xb[ib].qs     + 16 * iq + 4 * ir;
                device const half     * dh = &xb[ib].d;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i +  0] * (qs[i/2] & 0x0003);
                    acc2[0] += yl[i +  1] * (qs[i/2] & 0x0300);
                    acc1[1] += yl[i +  8] * (qs[i/2] & 0x000c);
                    acc2[1] += yl[i +  9] * (qs[i/2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (qs[i/2] & 0x0030);
                    acc2[2] += yl[i + 17] * (qs[i/2] & 0x3000);
                    acc1[3] += yl[i + 24] * (qs[i/2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (qs[i/2] & 0xc000);
                }
                float dall = (float)dh[0];
                float dmin = (float)dh[1] * (1.f/16.f);
                sumf += dall * ((acc1[0] + (1.f/256.f) * acc2[0]) * (sc[0] & 0xF) * (1.f/ 1.f) +
                                (acc1[1] + (1.f/256.f) * acc2[1]) * (sc[2] & 0xF) * (1.f/ 4.f) +
                                (acc1[2] + (1.f/256.f) * acc2[2]) * (sc[4] & 0xF) * (1.f/16.f) +
                                (acc1[3] + (1.f/256.f) * acc2[3]) * (sc[6] & 0xF) * (1.f/64.f))
                      - dmin * (sumy[0] * (sc[0] & 0xF0) +
                                sumy[1] * (sc[2] & 0xF0) +
                                sumy[2] * (sc[4] & 0xF0) +
                                sumy[3] * (sc[6] & 0xF0));
            }

            // Reduce across the simdgroup -> per-row dot for this slot.
            float dot = simd_sum(sumf);
            sum_slot += rw * dot;
        }

        acc_col[cs] = sum_slot;
    }

    if (lane == 0) {
        for (uint cs = 0; cs < COLS_PER_SG; cs++) {
            const uint d_out = col_base + sg * COLS_PER_SG + cs;
            if (d_out < D) {
                out[t * D + d_out] = half(float(residual[t * D + d_out]) + acc_col[cs]);
            }
        }
    }
}
