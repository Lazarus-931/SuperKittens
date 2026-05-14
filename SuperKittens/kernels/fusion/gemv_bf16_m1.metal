// gemv_bf16_m1.metal — M=1 bf16 GEMV (thread-per-output-col).
//
// Ported from temp/gemma4_kernel_lab/kernels.metal (lab_gemv_bf16_m1_threadcol).
// Validated to outperform the simdgroup-per-col variant on Apple M4 for the
// gemma4 decode-T=1 path. Used for QKV / O / PLE proj and the MLP `down` step.
//
// y[1, N] = x[1, K] @ W[K, N]   (row-major, K contiguous in x; W is K x N).
//
// Grid: (ceil(N/128), 1, 1) threadgroups; 128 threads per TG.
// Each thread owns one output column; cooperatively stages x into TG memory
// in BK=256 chunks, then does scalar accumulation.

#include <metal_stdlib>
using namespace metal;

[[host_name("gemv_bf16_m1")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemv_bf16_m1(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* W [[buffer(1)]],
    device       bfloat* y [[buffer(2)]],
    constant uint& N      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    constexpr uint BN = 128;
    constexpr uint BK = 256;
    const uint col0 = gid * BN;
    const uint col  = col0 + tid;
    threadgroup bfloat xs[BK];
    float acc = 0.0f;
    const uint K_main = (K / BK) * BK;
    for (uint k0 = 0; k0 < K_main; k0 += BK) {
        for (uint i = tid; i < BK / 2; i += BN) {
            reinterpret_cast<threadgroup bfloat2*>(xs)[i] =
                reinterpret_cast<const device bfloat2*>(x + k0)[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (col < N) {
            for (uint kk = 0; kk < BK; kk += 8) {
                float a0 = (float)W[(k0+kk+0)*N + col] * (float)xs[kk+0];
                float a1 = (float)W[(k0+kk+1)*N + col] * (float)xs[kk+1];
                float a2 = (float)W[(k0+kk+2)*N + col] * (float)xs[kk+2];
                float a3 = (float)W[(k0+kk+3)*N + col] * (float)xs[kk+3];
                float a4 = (float)W[(k0+kk+4)*N + col] * (float)xs[kk+4];
                float a5 = (float)W[(k0+kk+5)*N + col] * (float)xs[kk+5];
                float a6 = (float)W[(k0+kk+6)*N + col] * (float)xs[kk+6];
                float a7 = (float)W[(k0+kk+7)*N + col] * (float)xs[kk+7];
                acc += (a0+a1+a2+a3+a4+a5+a6+a7);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (col < N) {
        for (uint k = K_main; k < K; ++k) {
            acc += (float)x[k] * (float)W[k * N + col];
        }
        y[col] = bfloat(acc);
    }
}
