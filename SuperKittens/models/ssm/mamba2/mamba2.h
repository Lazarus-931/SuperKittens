//
//  mamba2.h — C bindings for Mamba-2 kernels
//

#ifndef SK_MAMBA2_H
#define SK_MAMBA2_H

#include <cstdint>

extern "C" {

// SSD / step are dispatched by the launcher (models/ssm/mamba2/launcher.c++,
// dispatch_layer) using the HF-correct mamba2_ssd / mamba2_step_ref signature,
// not via standalone C stubs.

int sk_conv1d_silu(void* x, void* weight, void* bias, void* y,
                   uint32_t B, uint32_t L, uint32_t C);

int sk_gate_norm(void* ssm_out, void* z, void* weight, void* y,
                 uint32_t B, uint32_t L, uint32_t E, float eps);

} // extern "C"

#endif
