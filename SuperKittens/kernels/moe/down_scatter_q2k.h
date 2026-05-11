#ifndef SK_MOE_DOWN_SCATTER_Q2K_H
#define SK_MOE_DOWN_SCATTER_Q2K_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Fused MoE down + routing-weight scale + residual add, with Q2_K-quantized
// expert weights. ~1.9× faster than fp16 moe_down_scatter at decode shapes;
// W_down is 6.1× smaller (2.625 bits/wt vs 16). Compute-bound on dequant.
//   hidden    : (T, top_k, N_int)               fp16
//   W_down    : (E, D, N_int/256) block_q2_K    Q2_K weights (84 bytes/block)
//   exp_ids   : (T, top_k)                      int32
//   route_w   : (T, top_k)                      fp16
//   residual  : (T, D)                          fp16
//   out       : (T, D)                          fp16
int sk_moe_down_scatter_q2k(void* hidden, void* W_down, void* exp_ids,
                            void* route_w, void* residual, void* out,
                            uint32_t T, uint32_t top_k, uint32_t E,
                            uint32_t D, uint32_t N_int);

#ifdef __cplusplus
}
#endif
#endif
