//
//  mamba3_ssm.metal — Mamba-3 SSM (optimized)
//  V cached in threadgroup, half4 output, half4 inter.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

[[host_name("mamba3_ssm")]]
[[kernel]]
void mamba3_ssm(
    device const half* Q, device const half* K, device const half* V,
    device const half* A, device const half* B, device const half* angle,
    device half* O,
    constant uint& L, constant uint& DQ, constant uint& DV, constant uint& CS,
    uint lid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint2 gid [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x, T = 128, HD = DQ / 2, nC = (L + CS - 1) / CS;
    const float PI = 3.141592653589793f;
    const size_t sQ = (size_t)bh * L * DQ, sV = (size_t)bh * L * DV, s1 = L, sA = L * HD;

    threadgroup half  Qc[32*64], Kc[32*64];  // Kc reused for Vc after state update
    threadgroup float Ac[32], Bc[32], a_cs[32], b_s[32];
    threadgroup half  ang[32*32];
    threadgroup float kv_s[64*64];
    threadgroup half  sc[32*32];  // half saves 2KB vs float

    for (uint i = lid; i < DQ * DV; i += T) kv_s[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ci = 0; ci < nC; ci++) {
        uint s = ci * CS, cl = min(CS, L - s);

        // Load Q, K, A, B, angle (V loaded later into Kc memory)
        for (uint i = lid; i < cl*DQ; i+=T) { Qc[i]=Q[bh*sQ + s*DQ + i]; Kc[i]=K[bh*sQ + s*DQ + i]; }
        for (uint i = lid; i < cl; i+=T)    { Ac[i]=float(A[bh*s1 + s + i]); Bc[i]=float(B[bh*s1 + s + i]); }
        for (uint i = lid; i < cl*HD; i+=T)    ang[i]=angle[bh*sA + s*HD + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cumsum + b_scale
        if (lid == 0) { float cs=0; for(uint t=0;t<cl;t++){cs+=Ac[t];a_cs[t]=cs;} }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = lid; t < cl; t += T) b_s[t] = 1.0f + Bc[t] * metal::fast::exp(-a_cs[t]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rotary on Q and K
        for (uint i = lid; i < cl*HD; i += T) {
            uint t=i/HD, p=i%HD; float th=a_cs[t]*float(ang[i])*PI;
            float cs=metal::fast::cos(th), sn=metal::fast::sin(th);
            float q0=float(Qc[t*DQ+p]), q1=float(Qc[t*DQ+p+HD]);
            Qc[t*DQ+p]=half(q0*cs - q1*sn); Qc[t*DQ+p+HD]=half(q0*sn + q1*cs);
            float k0=float(Kc[t*DQ+p]), k1=float(Kc[t*DQ+p+HD]);
            Kc[t*DQ+p]=half(k0*cs - k1*sn); Kc[t*DQ+p+HD]=half(k0*sn + k1*cs);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // State update: kv *= decay + K^T @ V (half4, V from threadgroup)
        float cd = metal::fast::exp(a_cs[cl-1]) * b_s[cl-1];
        for (uint j = lid; j < DV; j += T) {
            for (uint i = 0; i < DQ; i += 4) {
                float4 acc = float4(kv_s[(i+0)*DV+j],kv_s[(i+1)*DV+j],kv_s[(i+2)*DV+j],kv_s[(i+3)*DV+j]) * cd;
                for (uint t = 0; t < cl; t++) {
                    half4 k4 = reinterpret_cast<threadgroup half4*>(Kc + t*DQ)[i/4];
                    acc += float4(k4) * float(V[bh*sV + s*DV + t*DV + j]);
                }
                kv_s[(i+0)*DV+j]=acc.x; kv_s[(i+1)*DV+j]=acc.y; kv_s[(i+2)*DV+j]=acc.z; kv_s[(i+3)*DV+j]=acc.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores via simdgroup matmul: Q @ K^T
        const uint MR = cl / 8, MC = cl / 8;
        for (uint i = lid; i < cl*cl; i += T) sc[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < DQ / 8; k++) {
            if (simd < MR * MC) {
                uint r = simd / MC, c = simd % MC;
                simdgroup_half8x8 a, b;
                simdgroup_load(a, Qc + (r*8)*DQ + k*8, DQ);
                simdgroup_load(b, Kc + (c*8)*DQ + k*8, DQ, ulong2(0,0), true);

                simdgroup_float8x8 acc;
                // Load sc as half, convert to float
                for (uint ii = 0; ii < 8; ii++)
                    for (uint jj = 0; jj < 8; jj++)
                        acc.thread_elements()[ii*8+jj] = float(sc[(r*8+ii)*cl + c*8+jj]);

                simdgroup_multiply_accumulate(acc, a, b, acc);

                float2 v = reinterpret_cast<thread float2&>(acc.thread_elements());
                uint qid = lane / 4;
                uint lr = (qid & 4) + ((lane / 2) % 4);
                uint lc = (qid & 2) * 2 + (lane % 2) * 2;
                sc[(r*8+lr)*cl + c*8+lc] = half(v.x);
                sc[(r*8+lr)*cl + c*8+lc+1] = half(v.y);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Decay + causal mask (sc is half, convert for math)
        for (uint i = lid; i < cl*cl; i += T) {
            uint row = i / cl, col = i % cl;
            float val = float(sc[i]);
            if (col <= row) val *= metal::fast::exp(a_cs[row] - a_cs[col]);
            else val = 0.0f;
            sc[i] = half(val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V into Kc memory (Kc no longer needed after state update)
        threadgroup half* Vc = Kc;  // reuse Kc memory for V
        for (uint i = lid; i < cl*DV; i += T) Vc[i] = V[bh*sV + s*DV + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Output: intra + inter, half4 V reads
        for (uint pos = lid; pos < cl; pos += T) {
            float qd = metal::fast::exp(a_cs[pos]) * b_s[pos];
            for (uint j = 0; j < DV; j += 4) {
                float4 intra = float4(0);
                for (uint cc = 0; cc <= pos; cc++) {
                    float w = float(sc[pos*cl + cc]);
                    half4 v4 = reinterpret_cast<threadgroup half4*>(Vc + cc*DV)[j/4];
                    intra += w * float4(v4);
                }
                float4 inter = float4(0);
                for (uint di = 0; di < DQ; di += 4) {
                    half4 q4 = reinterpret_cast<threadgroup half4*>(Qc + pos*DQ)[di/4];
                    inter += float4(q4) * float4(kv_s[(di+0)*DV+j], kv_s[(di+1)*DV+j], kv_s[(di+2)*DV+j], kv_s[(di+3)*DV+j]);
                }
                float4 out4 = intra + qd * inter;
                reinterpret_cast<device half4*>(O + bh*sV + s*DV + pos*DV)[j/4] = half4(out4);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
