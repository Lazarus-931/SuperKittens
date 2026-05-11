#ifndef SK_MOE_SWIGLU_PAIR_H
#define SK_MOE_SWIGLU_PAIR_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Fused MoE swiglu pair (gate + up + SiLU + mul) — fp16, no quantization.
//   x        : (T, D)              fp16
//   W_gate   : (E, N_int, D)       fp16
//   W_up     : (E, N_int, D)       fp16
//   exp_ids  : (T, top_k)          int32
//   out      : (T, top_k, N_int)   fp16
int sk_moe_swiglu_pair(void* x, void* W_gate, void* W_up,
                       void* exp_ids, void* out,
                       uint32_t T, uint32_t top_k, uint32_t E,
                       uint32_t D, uint32_t N_int);

#ifdef __cplusplus
}
#endif
#endif
