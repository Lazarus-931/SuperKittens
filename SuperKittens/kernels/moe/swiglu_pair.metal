//
//  swiglu_pair.metal — fused MoE swiglu-pair matvec (gate + up + SiLU + mul).
//
//  For each (token, expert_slot), routes through expert E = exp_ids[t, slot]
//  and computes:
//      g = x[t]   @ W_gate[E]     // (D,) · (D, N_int) → (N_int,)
//      u = x[t]   @ W_up[E]
//      out[t, slot] = silu(g) * u
//
//  Inspired by ds4's `moe_mul_mv_id_iq2_xxs_pair_swiglu`. This is the fp16,
//  unquantized variant; same fusion shape. Decode-friendly (T=1) but works
//  for arbitrary T.
//
//  Layout:
//    x       : (T, D)                 fp16
//    W_gate  : (E, N_int, D)          fp16, row-major in D
//    W_up    : (E, N_int, D)          fp16
//    exp_ids : (T, top_k)             int32
//    out     : (T, top_k, N_int)      fp16
//
//  Tile: 256 threads, 8 simdgroups. Each TG handles 16 output cols of N_int
//  for one (token, slot). x[token] is staged into threadgroup memory in
//  X_BLOCK chunks; D may be larger than X_BLOCK (chunked accumulation).
//

#include <metal_stdlib>
using namespace metal;

constant constexpr uint COLS_PER_SG = 2;
constant constexpr uint N_SG        = 8;
constant constexpr uint COLS_PER_TG = COLS_PER_SG * N_SG;   // 16
constant constexpr uint TGSZ        = 32 * N_SG;            // 256
constant constexpr uint X_BLOCK     = 2048;                 // 4 KB shmem

[[host_name("moe_swiglu_pair")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void moe_swiglu_pair(
    device const half*  x         [[buffer(0)]],
    device const half*  W_gate    [[buffer(1)]],
    device const half*  W_up      [[buffer(2)]],
    device const int*   exp_ids   [[buffer(3)]],
    device half*        out       [[buffer(4)]],
    constant uint&      T         [[buffer(5)]],
    constant uint&      top_k     [[buffer(6)]],
    constant uint&      D         [[buffer(7)]],
    constant uint&      N_int     [[buffer(8)]],
    uint3   gid                   [[threadgroup_position_in_grid]],
    ushort  sg                    [[simdgroup_index_in_threadgroup]],
    ushort  lane                  [[thread_index_in_simdgroup]])
{
    const uint t_slot   = gid.y;
    const uint t        = t_slot / top_k;
    const uint slot     = t_slot % top_k;
    const uint col_base = gid.x * COLS_PER_TG;
    if (t >= T) return;

    const int  eid    = exp_ids[t * top_k + slot];
    const uint base_w = (uint)eid * N_int * D;
    const uint tid    = sg * 32 + lane;

    threadgroup half xs[X_BLOCK];

    // Per-thread accumulators for this simdgroup's 2 output cols.
    float g_acc[COLS_PER_SG] = {0.f, 0.f};
    float u_acc[COLS_PER_SG] = {0.f, 0.f};

    for (uint k0 = 0; k0 < D; k0 += X_BLOCK) {
        const uint chunk = min((uint)X_BLOCK, D - k0);

        // Coop-load x[t, k0:k0+chunk] into shmem.
        for (uint i = tid; i < chunk; i += TGSZ) {
            xs[i] = x[t * D + k0 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each simdgroup contributes to its 2 cols.
        for (uint cs = 0; cs < COLS_PER_SG; cs++) {
            const uint col = col_base + sg * COLS_PER_SG + cs;
            if (col >= N_int) continue;
            const device half* wg = W_gate + base_w + col * D + k0;
            const device half* wu = W_up   + base_w + col * D + k0;

            float g = 0.f, u = 0.f;
            for (uint k = lane; k < chunk; k += 32) {
                float xv = float(xs[k]);
                g += xv * float(wg[k]);
                u += xv * float(wu[k]);
            }
            g_acc[cs] += g;
            u_acc[cs] += u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce + activate + write.
    for (uint cs = 0; cs < COLS_PER_SG; cs++) {
        const uint col = col_base + sg * COLS_PER_SG + cs;
        if (col >= N_int) continue;
        const float g = simd_sum(g_acc[cs]);
        const float u = simd_sum(u_acc[cs]);
        if (lane == 0) {
            const float silu_g = g / (1.f + metal::precise::exp(-g));
            out[t_slot * N_int + col] = half(silu_g * u);
        }
    }
}
