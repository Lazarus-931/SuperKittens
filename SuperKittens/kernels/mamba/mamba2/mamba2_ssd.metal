//
//  mamba2_ssd.metal
//  SuperKittens — Mamba-2 selective scan
//
//  128 threads, half4 vectorized inner loops, 2 barriers/row.

#include <metal_stdlib>
using namespace metal;

[[host_name("mamba2_ssd")]]
[[kernel]]
void mamba2_ssd(
    device const half* Q,
    device const half* K,
    device const half* V,
    device const half* A_log,
    device half* y,
    constant uint& L, constant uint& Ds, constant uint& Dv, constant uint& H,
    uint  lid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint2 gid  [[threadgroup_position_in_grid]])
{
    const uint b = gid.x, head = gid.y;
    if (head >= H) return;

    const size_t Ls  = (size_t)H * Ds;
    const size_t Lv  = (size_t)H * Dv;
    const size_t La  = (size_t)H;
    const size_t qk_bo = (size_t)b * L * H * Ds + (size_t)head * Ds;
    const size_t v_bo  = (size_t)b * L * H * Dv + (size_t)head * Dv;
    const size_t a_bo  = (size_t)b * L * H + (size_t)head;

    threadgroup float h_state[64 * 64];
    threadgroup half  k_row[64], v_row[64], q_row[64];

    for (uint i = lid; i < Ds * Dv; i += 128) h_state[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = 0; pos < L; pos++) {
        float decay = metal::fast::exp(float(A_log[a_bo + pos * La]));
        const size_t qk_off = qk_bo + pos * Ls;
        const size_t v_off  = v_bo  + pos * Lv;

        // Load K, Q, V (scalar — only 64 elements each, cooperative load is fast)
        if (lid < Ds) { k_row[lid] = K[qk_off + lid]; q_row[lid] = Q[qk_off + lid]; }
        if (lid < Dv)   v_row[lid] = V[v_off + lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // State update: h = h*decay + K^T @ V (half4 vectorized k reads)
        for (uint hc = lid; hc < Dv; hc += 128) {
            float vj = float(v_row[hc]);
            for (uint hi = 0; hi < Ds; hi += 4) {
                half4 kv = reinterpret_cast<threadgroup half4*>(k_row)[hi / 4];
                uint base = hi * Dv + hc;
                h_state[base + 0*Dv] = h_state[base + 0*Dv] * decay + float(kv.x) * vj;
                h_state[base + 1*Dv] = h_state[base + 1*Dv] * decay + float(kv.y) * vj;
                h_state[base + 2*Dv] = h_state[base + 2*Dv] * decay + float(kv.z) * vj;
                h_state[base + 3*Dv] = h_state[base + 3*Dv] * decay + float(kv.w) * vj;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Output: y = Q @ h (half4 vectorized q reads)
        for (uint hc = lid; hc < Dv; hc += 128) {
            float acc = 0;
            for (uint hi = 0; hi < Ds; hi += 4) {
                half4 qv = reinterpret_cast<threadgroup half4*>(q_row)[hi / 4];
                acc += float(qv.x) * h_state[(hi+0)*Dv + hc];
                acc += float(qv.y) * h_state[(hi+1)*Dv + hc];
                acc += float(qv.z) * h_state[(hi+2)*Dv + hc];
                acc += float(qv.w) * h_state[(hi+3)*Dv + hc];
            }
            y[v_off + hc] = half(acc);
        }
    }
}
