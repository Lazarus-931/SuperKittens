#ifndef SK_MOE_SWIGLU_PAIR_IQ2XXS_H
#define SK_MOE_SWIGLU_PAIR_IQ2XXS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Fused MoE swiglu pair with IQ2_XXS-quantized expert weights.
//   x        : (T, D)                            fp16
//   W_gate   : (E, N_int, D/256) block_iq2_xxs   ~2.06 bits/weight
//   W_up     : (E, N_int, D/256) block_iq2_xxs
//   exp_ids  : (T, top_k)                        int32
//   out      : (T, top_k, N_int)                 fp16
// Computes silu(W_gate[id] · x) * (W_up[id] · x).
// ~1.4× faster than fp16 swiglu_pair at decode shapes; 7.76× smaller weights.
int sk_moe_swiglu_pair_iq2xxs(void* x, void* W_gate, void* W_up,
                              void* exp_ids, void* out,
                              uint32_t T, uint32_t top_k, uint32_t E,
                              uint32_t D, uint32_t N_int);

#ifdef __cplusplus
}
#endif
#endif
