//
//  mamba2.h — C bindings for Mamba-2 kernels
//

#ifndef SK_MAMBA2_H
#define SK_MAMBA2_H

#include <cstdint>

extern "C" {

int sk_mamba2_ssd(void* Q, void* K, void* V, void* A_log, void* y,
                  uint32_t B, uint32_t L, uint32_t Ds, uint32_t Dv, uint32_t H);

int sk_conv1d_silu(void* x, void* weight, void* bias, void* y,
                   uint32_t B, uint32_t L, uint32_t C);

int sk_gate_norm(void* ssm_out, void* z, void* weight, void* y,
                 uint32_t B, uint32_t L, uint32_t E, float eps);

} // extern "C"

#endif
