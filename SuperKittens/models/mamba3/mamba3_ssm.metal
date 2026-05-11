//
//  mamba3_ssm_v2.metal — DV-split Mamba-3 SSM
//
//  Strategy: grid (BH, DV/BV, 1), each threadgroup owns one DV slice of width BV.
//  This raises grid occupancy from BH to BH*(DV/BV) — M2's 10 GPU cores were starved
//  by v1's (BH,1,1) grid. Scores (Q@K^T) and rotary are recomputed per TG; cheap
//  relative to the output loop.
//
//  Threadgroup state: kv_s is now (DQ, BV) floats — 128*64*4 = 32KB.
//  V is loaded per chunk into Vc (CS*BV*half = 4KB for CS=32, BV=64).
//
//  Same numerics as v1 (uses simdgroup matmul for scores).
//
//  Constraints expected at call sites here:
//    DQ % 8 == 0, DV % BV == 0, CS in {16, 32}.
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BV = 32;   // DV tile per threadgroup
constant constexpr uint T  = 128;  // threads per threadgroup
constant constexpr uint CS_MAX = 32;
constant constexpr uint DQ_MAX = 64;

[[host_name("mamba3_ssm")]]
[[kernel, max_total_threads_per_threadgroup(T)]]
void mamba3_ssm(
    device const half* Q, device const half* K, device const half* V,
    device const half* A, device const half* B, device const half* angle,
    device half* O,
    constant uint& L  [[buffer(7)]],
    constant uint& DQ [[buffer(8)]],
    constant uint& DV [[buffer(9)]],
    constant uint& CS [[buffer(10)]],
    device const float* h_state_in  [[buffer(11)]],
    device float*       h_state_out [[buffer(12)]],
    device const float* a_cs_in     [[buffer(13)]],
    device float*       a_cs_out    [[buffer(14)]],
    constant uint& state_flags      [[buffer(15)]],
    uint lid  [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]])
{
    const uint bh   = gid.x;
    const uint vtg  = gid.y;            // which DV tile (0..DV/BV-1)
    const uint v_off= vtg * BV;
    const uint HD   = DQ / 2;
    const uint nC   = (L + CS - 1) / CS;
    const float PI  = 3.141592653589793f;

    const size_t sQ = (size_t)bh * L * DQ;
    const size_t sV = (size_t)bh * L * DV;
    const size_t s1 = (size_t)bh * L;
    const size_t sA = (size_t)bh * L * HD;

    // shared
    threadgroup half  Qc[CS_MAX * DQ_MAX];   // 32*128 = 8KB
    threadgroup half  Kc[CS_MAX * DQ_MAX];   // 8KB
    threadgroup half  Vc[CS_MAX * BV];       // 32*64 = 4KB
    threadgroup half  ang[CS_MAX * 32];      // 32*32 = 2KB  (HD <= 32 since DQ <= 64)
    threadgroup float Ac[CS_MAX], Bc[CS_MAX], a_cs[CS_MAX], b_s[CS_MAX];
    threadgroup half  sc[CS_MAX * CS_MAX];   // 32*32 = 2KB
    threadgroup float kv_s[DQ_MAX * BV];     // 128*64 = 32KB
    threadgroup float a_cs_off_tg[1];

    // Load / zero kv tile from global state (DV-sliced)
    const size_t state_bo = (size_t)bh * DQ * DV;  // h_state layout (BH,DQ,DV)
    if (state_flags & 1u) {
        for (uint i = lid; i < DQ * BV; i += T) {
            uint qi = i / BV, vi = i % BV;
            kv_s[i] = h_state_in[state_bo + qi * DV + v_off + vi];
        }
        if (lid == 0) a_cs_off_tg[0] = a_cs_in[bh];
    } else {
        for (uint i = lid; i < DQ * BV; i += T) kv_s[i] = 0.0f;
        if (lid == 0) a_cs_off_tg[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a_cs_off = a_cs_off_tg[0];

    for (uint ci = 0; ci < nC; ci++) {
        uint s = ci * CS, cl = min(CS, L - s);

        // load Q, K (full DQ), V tile (BV columns), A, B, angle
        for (uint i = lid; i < cl * DQ; i += T) {
            Qc[i] = Q[sQ + s*DQ + i];
            Kc[i] = K[sQ + s*DQ + i];
        }
        for (uint i = lid; i < cl * BV; i += T) {
            uint t = i / BV, vi = i % BV;
            Vc[i] = V[sV + (s + t) * DV + v_off + vi];
        }
        for (uint i = lid; i < cl; i += T) {
            Ac[i] = float(A[s1 + s + i]);
            Bc[i] = float(B[s1 + s + i]);
        }
        for (uint i = lid; i < cl * HD; i += T) ang[i] = angle[sA + s*HD + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // cumsum + b_scale (lid 0)
        if (lid == 0) {
            float cs = 0;
            for (uint t = 0; t < cl; t++) { cs += Ac[t]; a_cs[t] = cs; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = lid; t < cl; t += T) b_s[t] = 1.0f + Bc[t] * metal::fast::exp(-a_cs[t]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // rotary on Q, K (still per-token over HD)
        for (uint i = lid; i < cl * HD; i += T) {
            uint t = i / HD, p = i % HD;
            float th = (a_cs[t] + a_cs_off) * float(ang[i]) * PI;
            float cs_ = metal::fast::cos(th), sn = metal::fast::sin(th);
            float q0 = float(Qc[t*DQ + p]),     q1 = float(Qc[t*DQ + p + HD]);
            Qc[t*DQ + p]      = half(q0*cs_ - q1*sn);
            Qc[t*DQ + p + HD] = half(q0*sn  + q1*cs_);
            float k0 = float(Kc[t*DQ + p]),     k1 = float(Kc[t*DQ + p + HD]);
            Kc[t*DQ + p]      = half(k0*cs_ - k1*sn);
            Kc[t*DQ + p + HD] = half(k0*sn  + k1*cs_);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── state update on the BV-wide tile ──
        // kv[i, j] *= cd + sum_t K[t,i] * V[t,j]   for j in [v_off, v_off+BV), i in [0,DQ)
        float cd = metal::fast::exp(a_cs[cl-1]) * b_s[cl-1];
        // distribute the (DQ * BV) updates across T threads
        for (uint idx = lid; idx < DQ * BV; idx += T) {
            uint i = idx / BV, j = idx % BV;
            float acc = kv_s[i * BV + j] * cd;
            for (uint t = 0; t < cl; t++) {
                acc += float(Kc[t*DQ + i]) * float(Vc[t*BV + j]);
            }
            kv_s[i * BV + j] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── scores Q @ K^T via simdgroup mma (same as v1) ──
        const uint MR = cl / 8, MC = cl / 8;
        for (uint i = lid; i < cl * cl; i += T) sc[i] = 0.0h;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < DQ / 8; k++) {
            if (simd < MR * MC) {
                uint r = simd / MC, c = simd % MC;
                simdgroup_half8x8 a, b;
                simdgroup_load(a, Qc + (r*8)*DQ + k*8, DQ);
                simdgroup_load(b, Kc + (c*8)*DQ + k*8, DQ, ulong2(0,0), true);

                simdgroup_float8x8 acc;
                for (uint ii = 0; ii < 8; ii++)
                    for (uint jj = 0; jj < 8; jj++)
                        acc.thread_elements()[ii*8+jj] = float(sc[(r*8+ii)*cl + c*8+jj]);
                simdgroup_multiply_accumulate(acc, a, b, acc);
                float2 v = reinterpret_cast<thread float2&>(acc.thread_elements());
                uint qid = lane / 4;
                uint lr = (qid & 4) + ((lane / 2) % 4);
                uint lc = (qid & 2) * 2 + (lane % 2) * 2;
                sc[(r*8+lr)*cl + c*8+lc]     = half(v.x);
                sc[(r*8+lr)*cl + c*8+lc + 1] = half(v.y);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // decay + causal mask
        for (uint i = lid; i < cl * cl; i += T) {
            uint row = i / cl, col = i % cl;
            float val = float(sc[i]);
            if (col <= row) val *= metal::fast::exp(a_cs[row] - a_cs[col]);
            else val = 0.0f;
            sc[i] = half(val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── output: intra (sc @ V) + inter (Q @ kv_s), BV columns ──
        // distribute (cl * BV) outputs over T threads
        for (uint idx = lid; idx < cl * BV; idx += T) {
            uint pos = idx / BV, j = idx % BV;
            float qd = metal::fast::exp(a_cs[pos]) * b_s[pos];
            float intra = 0.0f;
            for (uint cc = 0; cc <= pos; cc++) {
                intra += float(sc[pos*cl + cc]) * float(Vc[cc*BV + j]);
            }
            float inter = 0.0f;
            for (uint di = 0; di < DQ; di++) {
                inter += float(Qc[pos*DQ + di]) * kv_s[di * BV + j];
            }
            float out_v = intra + qd * inter;
            O[sV + (s + pos) * DV + v_off + j] = half(out_v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a_cs_off += a_cs[cl-1];
    }

    if (state_flags & 2u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < DQ * BV; i += T) {
            uint qi = i / BV, vi = i % BV;
            h_state_out[state_bo + qi * DV + v_off + vi] = kv_s[i];
        }
        // only one tile writes a_cs_out (matches v1 behavior)
        if (lid == 0 && vtg == 0) a_cs_out[bh] = a_cs_off;
    }
}
