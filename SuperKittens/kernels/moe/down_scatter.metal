//  down_scatter_v2.metal — vectorized + tiled fused MoE down-proj.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint COLS_PER_SG = 2;
constant constexpr uint N_SG        = 8;
constant constexpr uint COLS_PER_TG = COLS_PER_SG * N_SG;   // 16
constant constexpr uint TGSZ        = 32 * N_SG;            // 256
constant constexpr uint MAX_TOPK    = 16;
constant constexpr uint CHUNK_H     = 128;                  // halves per tile (CHUNK_H4 == simdgroup width 32, zero-tail at n_int multiples of 128)
constant constexpr uint CHUNK_H4    = CHUNK_H / 4;          // 32

[[host_name("moe_down_scatter")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void moe_down_scatter_v2(
    device const half*  hidden    [[buffer(0)]],
    device const half*  W_down    [[buffer(1)]],
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

    const uint N_int_h4 = N_int / 4;
    // Round up so a partial tail tile (n_int % CHUNK_H != 0) is not silently dropped.
    const uint n_tiles  = (N_int_h4 + CHUNK_H4 - 1) / CHUNK_H4;

    // top_k * CHUNK_H4 half4 = top_k * 128 half4. At top_k=16 → 2048 half4 = 16KB.
    threadgroup half4 hidden_tg[MAX_TOPK * CHUNK_H4];
    threadgroup int   exp_tg[MAX_TOPK];
    threadgroup half  rw_tg[MAX_TOPK];

    if (tid < top_k) {
        exp_tg[tid] = exp_ids[t * top_k + tid];
        rw_tg[tid]  = route_w[t * top_k + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-output-col accumulators (one per cs, kept in registers).
    float acc0 = 0.f;
    float acc1 = 0.f;

    const device half4* hidden_v_base = reinterpret_cast<const device half4*>(
        hidden + t * top_k * N_int);

    for (uint tile = 0; tile < n_tiles; tile++) {
        const uint tile_h4_off = tile * CHUNK_H4;

        // Cooperative load: top_k * CHUNK_H4 half4s.
        // Tail-tile lanes (tile_h4_off+off >= N_int_h4) zero-fill so they contribute 0 to the dot.
        const uint total = top_k * CHUNK_H4;
        for (uint i = tid; i < total; i += TGSZ) {
            const uint s   = i / CHUNK_H4;
            const uint off = i - s * CHUNK_H4;
            const uint h4_col = tile_h4_off + off;
            hidden_tg[i] = (h4_col < N_int_h4)
                ? hidden_v_base[s * N_int_h4 + h4_col]
                : half4(0.h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each sg handles COLS_PER_SG output cols.
        for (uint cs = 0; cs < COLS_PER_SG; cs++) {
            const uint d = col_base + sg * COLS_PER_SG + cs;
            if (d >= D) continue;

            float tile_dot = 0.f;
            for (uint s = 0; s < top_k; s++) {
                const int   eid = exp_tg[s];
                const float rw  = float(rw_tg[s]);

                const threadgroup half4* h4 = hidden_tg + s * CHUNK_H4;
                const device half4* w4 = reinterpret_cast<const device half4*>(
                    W_down + ((uint)eid * D + d) * N_int) + tile_h4_off;

                float dot = 0.f;
                // Mask weight reads past N_int_h4 in the tail tile so we don't OOB-read W_down.
                // (hidden_tg is already zero-filled past the tail above.)
                for (uint n = lane; n < CHUNK_H4; n += 32) {
                    if (tile_h4_off + n < N_int_h4) {
                        half4 hv = h4[n];
                        half4 wv = w4[n];
                        float4 p = float4(hv) * float4(wv);
                        dot += p.x + p.y + p.z + p.w;
                    }
                }
                dot = simd_sum(dot);
                tile_dot += rw * dot;
            }

            if (cs == 0) acc0 += tile_dot;
            else         acc1 += tile_dot;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write outputs (one per cs).
    if (lane == 0) {
        const uint d0 = col_base + sg * COLS_PER_SG + 0;
        const uint d1 = col_base + sg * COLS_PER_SG + 1;
        if (d0 < D) out[t * D + d0] = half(float(residual[t * D + d0]) + acc0);
        if (d1 < D) out[t * D + d1] = half(float(residual[t * D + d1]) + acc1);
    }
}
