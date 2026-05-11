//
//  mamba2_ssd_v2.metal
//  SuperKittens — Mamba-2 SSD v2: slice DV across threadgroups for higher occupancy.
//
//  Grid: (BH, DV/BV, 1). Threads/tg: 64. Each tg owns Ds rows x BV cols of state.
//  Hypothesis: v1 grid is (BH,1,1) ~ 32 tgs on a typical M-series core count;
//  slicing DV gives more parallel tgs and shrinks per-tg shmem 2x (Ds*BV*4 bytes).
//
//  Constraints: Ds <= 64, DV must be divisible by BV (BV=32).

#include <metal_stdlib>
using namespace metal;

constant uint BV = 32;        // state cols handled per threadgroup
constant uint DS_MAX = 64;    // max state-dim supported
constant uint TG_THREADS = 64;

[[host_name("mamba2_ssd")]]
[[kernel]]
void mamba2_ssd(
    device const half* Q       [[buffer(0)]],
    device const half* K       [[buffer(1)]],
    device const half* V       [[buffer(2)]],
    device const half* A_log   [[buffer(3)]],
    device half*       y       [[buffer(4)]],
    constant uint& L           [[buffer(5)]],
    constant uint& Ds          [[buffer(6)]],
    constant uint& Dv          [[buffer(7)]],
    constant uint& H           [[buffer(8)]],
    device const float* h_state_in  [[buffer(9)]],
    device float*       h_state_out [[buffer(10)]],
    constant uint& state_flags      [[buffer(11)]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x;
    const uint vc = gid.y;           // which DV chunk
    const uint v0 = vc * BV;

    const size_t qk_bo = (size_t)bh * L * Ds;
    const size_t v_bo  = (size_t)bh * L * Dv;
    const size_t a_bo  = (size_t)bh * L;

    // State slice: Ds rows x BV cols
    threadgroup float h_state[DS_MAX * BV];
    threadgroup half  k_row[DS_MAX], q_row[DS_MAX], v_row[BV];

    const size_t state_bo = (size_t)bh * Ds * Dv + (size_t)0;
    // Load / zero h state slice for this DV chunk
    if (state_flags & 1u) {
        for (uint i = lid; i < Ds * BV; i += TG_THREADS) {
            uint row = i / BV;
            uint col = i % BV;
            h_state[row * BV + col] = h_state_in[state_bo + row * Dv + v0 + col];
        }
    } else {
        for (uint i = lid; i < Ds * BV; i += TG_THREADS) h_state[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = 0; pos < L; pos++) {
        float decay = metal::fast::exp(float(A_log[a_bo + pos]));
        const size_t qk_off = qk_bo + (size_t)pos * Ds;
        const size_t v_off  = v_bo  + (size_t)pos * Dv;

        if (lid < Ds) {
            k_row[lid] = K[qk_off + lid];
            q_row[lid] = Q[qk_off + lid];
        }
        if (lid < BV) {
            v_row[lid] = V[v_off + v0 + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // h[hi, hc] = h[hi, hc] * decay + K[hi] * V[v0+hc]
        // Threads cover BV columns -> at TG=64, two cols per thread max (BV=32 => 1 col each w/ 32 idle)
        // We use lid < BV; each thread walks all Ds rows.
        if (lid < BV) {
            float vj = float(v_row[lid]);
            threadgroup half4 const* kv4 = reinterpret_cast<threadgroup half4 const*>(k_row);
            threadgroup half4 const* qv4 = reinterpret_cast<threadgroup half4 const*>(q_row);
            float acc = 0.0f;
            for (uint hi = 0; hi < Ds; hi += 4) {
                half4 kv = kv4[hi >> 2];
                half4 qv = qv4[hi >> 2];
                uint b0 = (hi + 0) * BV + lid;
                uint b1 = (hi + 1) * BV + lid;
                uint b2 = (hi + 2) * BV + lid;
                uint b3 = (hi + 3) * BV + lid;
                float h0 = h_state[b0] * decay + float(kv.x) * vj;
                float h1 = h_state[b1] * decay + float(kv.y) * vj;
                float h2 = h_state[b2] * decay + float(kv.z) * vj;
                float h3 = h_state[b3] * decay + float(kv.w) * vj;
                h_state[b0] = h0;
                h_state[b1] = h1;
                h_state[b2] = h2;
                h_state[b3] = h3;
                acc += float(qv.x) * h0;
                acc += float(qv.y) * h1;
                acc += float(qv.z) * h2;
                acc += float(qv.w) * h3;
            }
            y[v_off + v0 + lid] = half(acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (state_flags & 2u) {
        for (uint i = lid; i < Ds * BV; i += TG_THREADS) {
            uint row = i / BV;
            uint col = i % BV;
            h_state_out[state_bo + row * Dv + v0 + col] = h_state[row * BV + col];
        }
    }
}
