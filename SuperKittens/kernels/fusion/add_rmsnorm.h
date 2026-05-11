#ifndef SK_FUSION_ADD_RMSNORM_H
#define SK_FUSION_ADD_RMSNORM_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Fused residual-add + RMSNorm:
//   y[t, d]      = x[t, d] + delta[t, d]
//   y_norm[t, d] = y[t, d] / sqrt(mean(y[t]^2) + eps) * gamma[d]
//
// Replaces (add_f16 + rmsnorm). Wins 1.57–2.34× across decode and prefill
// shapes by saving one full d_model HBM round-trip per layer × 2 (pre-mlp +
// pre-next-attn norms). See results.md.
//
//   x, delta : (T, D)  fp16
//   gamma    : (D,)    fp16
//   y        : (T, D)  fp16  — sum, needed for next residual
//   y_norm   : (T, D)  fp16  — normalized, input to next op
int sk_add_rmsnorm(void* x, void* delta, void* gamma,
                   void* y, void* y_norm,
                   uint32_t T, uint32_t D, float eps);

#ifdef __cplusplus
}
#endif
#endif
