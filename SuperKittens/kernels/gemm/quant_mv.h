#ifndef SK_GEMM_QUANT_MV_H
#define SK_GEMM_QUANT_MV_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// fp16 activation × Q2_K-packed weight → fp16 output (matvec).
//   x : (D,)               fp16
//   W : (N, D / 256)       block_q2_K[] (84 bytes/block, 2.625 bits/weight)
//   y : (N,)               fp16
// D must be a multiple of 256. From ds4's kernel_mul_mv_q2_K_f32_impl,
// simplified for fp16 acts + single-row matvec. ~1.7× faster than fp16
// matvec at decode shapes (D=4096, N=8192) — weight bytes are 6× smaller.
int sk_gemm_q2k_mv(void* x, void* W, void* y, uint32_t D, uint32_t N);

// Same as above but Q4_K weights (~1.8× faster than fp16 matvec; weights
// 3.5× smaller — 4.5 bits/weight via 144-byte/256-weight block_q4_K).
int sk_gemm_q4k_mv(void* x, void* W, void* y, uint32_t D, uint32_t N);

// IQ2_XXS weights (~1.2× faster than fp16; weights ~8× smaller —
// ~2.06 bits/weight via 66-byte block_iq2_xxs with a 256-entry grid table
// baked into the kernel). Compute-bound on dequant, not bandwidth-bound.
int sk_gemm_iq2xxs_mv(void* x, void* W, void* y, uint32_t D, uint32_t N);

#ifdef __cplusplus
}
#endif
#endif
