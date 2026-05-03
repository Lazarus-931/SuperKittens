//
//  conv2d.metal — Direct convolution via im2col + simdgroup GEMM (fused)
//  Loads input window into threadgroup, extracts patches, does MMA.
//  TM=32 (4×8 spatial), TN=64 (all C_out), BK=64 (full C_in)
//  Individual constant params for best codegen. 16KB threadgroup.
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

enum : uint { H_TILE = 4, W_TILE = 8, TM = 32, TN = 64, BK = 64, MR = 2, MC = 8 };

[[host_name("conv2d_tiled")]]
[[kernel]]
void conv2d_tiled(
    device const half* x,
    device const half* weight,
    device const half* bias,
    device half* y,
    constant uint& N, constant uint& H, constant uint& W, constant uint& C_in,
    constant uint& K_H, constant uint& K_W, constant uint& C_out,
    constant uint& H_out, constant uint& W_out,
    uint3 gid [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint n  = gid.z;
    const uint h0 = gid.y * H_TILE;
    const uint w0 = gid.x * W_TILE;
    if (n >= N) return;

    // Max window for 3×3 kernel: (4+2)*(8+2)*64 = 3840 halfs ≈ 7.5KB
    enum : uint { MAX_WIN = (H_TILE + 2) * (W_TILE + 2) * BK };

    threadgroup half win  [MAX_WIN];
    threadgroup half As   [TM * BK];     // input patch:  32*64  = 2048 halfs ≈ 4KB
    threadgroup half Bs   [BK * TN];     // weight slice: 64*64  = 4096 halfs ≈ 8KB
    // total ≈ 19.5KB

    const uint n_a = TM * BK;
    const uint n_b = BK * TN;
    const uint tid = simd * 32 + lane;

    simdgroup_float8x8 acc[MR][MC] = {};

    const device half4* x4 = reinterpret_cast<const device half4*>(x);
    const device half4* w4 = reinterpret_cast<const device half4*>(weight);

    // Load input window (once — all kernel positions use it)
    const uint H_WIN = H_TILE + K_H - 1;
    const uint W_WIN = W_TILE + K_W - 1;
    for (uint i = tid; i < MAX_WIN / 4; i += 64) {
        uint r = (i * 4) / BK;
        uint c = (i * 4) % BK;
        uint wh = r / W_WIN, ww = r % W_WIN;
        uint hi = h0 + wh, wi = w0 + ww;
        reinterpret_cast<threadgroup half4*>(win)[i] =
            (hi < H && wi < W)
                ? x4[(((n * H + hi) * W + wi) * C_in + c) / 4]
                : half4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kh = 0; kh < K_H; kh++) {
        for (uint kw = 0; kw < K_W; kw++) {

            // Extract As from input window at offset (kh, kw)
            for (uint i = tid; i < n_a / 4; i += 64) {
                uint r = (i * 4) / BK;
                uint c = (i * 4) % BK;
                uint r_h = r / W_TILE, r_w = r % W_TILE;
                uint wh = r_h + kh, ww = r_w + kw;
                uint win_off = (wh * W_WIN + ww) * BK + c;
                reinterpret_cast<threadgroup half4*>(As)[i] =
                    reinterpret_cast<threadgroup half4*>(win)[win_off / 4];
            }

            // Load weight slice Bs [BK, TN] — weight[kh][kw][:][:]
            for (uint i = tid; i < n_b / 4; i += 64) {
                uint r = (i * 4) / TN;
                uint c = (i * 4) % TN;
                reinterpret_cast<threadgroup half4*>(Bs)[i] =
                    (r < C_in && c < C_out)
                        ? w4[(((kh * K_W + kw) * C_in + r) * C_out + c) / 4]
                        : half4(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // MMA: As[TM][BK] @ Bs[BK][TN]
            const uint r0 = simd * 16;
            for (uint k = 0; k < BK / 8; k++) {
                for (uint r = 0; r < MR; r++) {
                    simdgroup_half8x8 a;
                    simdgroup_load(a, As + (r0 + r * 8) * BK + k * 8, BK);
                    for (uint c = 0; c < MC; c++) {
                        simdgroup_half8x8 b;
                        simdgroup_load(b, Bs + k * 8 * TN + c * 8, TN);
                        simdgroup_multiply_accumulate(acc[r][c], a, b, acc[r][c]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store acc → device directly (skip Cs threadgroup buffer)
    device half4* y4 = reinterpret_cast<device half4*>(y);
    const device half4* b4 = reinterpret_cast<const device half4*>(bias);
    const uint r0 = simd * 16;

    for (uint r = 0; r < MR; r++) {
        for (uint c = 0; c < MC; c++) {
            float2 v = reinterpret_cast<thread float2&>(acc[r][c].thread_elements());
            uint qid = lane / 4;
            uint lr = (qid & 4) + ((lane / 2) % 4);
            uint lc = (qid & 2) * 2 + (lane % 2) * 2;

            uint row  = r0 + r * 8 + lr;
            uint col  = c * 8 + lc;
            uint r_h  = row / W_TILE, r_w = row % W_TILE;
            uint h_out = h0 + r_h, w_out = w0 + r_w;

            if (h_out < H_out && w_out < W_out && col < C_out) {
                half val0 = half(v.x);
                half val1 = half(v.y);
                if (bias) {
                    val0 = half(float(val0) + float((reinterpret_cast<const device half*>(bias))[col]));
                    val1 = half(float(val1) + float((reinterpret_cast<const device half*>(bias))[col + 1]));
                }
                uint y_off = (((n * H_out + h_out) * W_out + w_out) * C_out + col);
                (reinterpret_cast<device half*>(y))[y_off]     = val0;
                (reinterpret_cast<device half*>(y))[y_off + 1] = val1;
            }
        }
    }
}
