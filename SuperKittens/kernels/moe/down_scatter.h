#ifndef SK_MOE_DOWN_SCATTER_H
#define SK_MOE_DOWN_SCATTER_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Fused MoE down-proj + routing-weight scale + residual add.
//   hidden    : (T, top_k, N_int)   fp16
//   W_down    : (E, D, N_int)       fp16
//   exp_ids   : (T, top_k)          int32
//   route_w   : (T, top_k)          fp16
//   residual  : (T, D)              fp16
//   out       : (T, D)              fp16
int sk_moe_down_scatter(void* hidden, void* W_down, void* exp_ids,
                        void* route_w, void* residual, void* out,
                        uint32_t T, uint32_t top_k, uint32_t E,
                        uint32_t D, uint32_t N_int);

#ifdef __cplusplus
}
#endif
#endif
