//
//  mamba2.h — C bindings for Mamba-2 kernels
//

#ifndef SK_MAMBA2_H
#define SK_MAMBA2_H

#include <cstdint>

extern "C" {

int sk_mamba2_ssd(void* Q, void* K, void* V, void* A_log, void* y,
                  uint32_t B, uint32_t L, uint32_t Ds, uint32_t Dv, uint32_t H);

// Variant with optional persistent SSM state (fp32, shape (B*H, Ds, Dv)).
// h_state_in == NULL  → start from zeros.   h_state_out == NULL → discard final state.
int sk_mamba2_ssd_state(void* Q, void* K, void* V, void* A_log, void* y,
                        void* h_state_in, void* h_state_out,
                        uint32_t B, uint32_t L, uint32_t Ds, uint32_t Dv, uint32_t H);

// Single-token decode step. Per-token shapes:
//   x_t:     (BH, Dv) fp16     B_t: (BH, Ds) fp16     C_t: (BH, Ds) fp16
//   A_log_t: (BH,) fp16        y_t: (BH, Dv) fp16
//   h_state: (BH, Ds, Dv) fp32 — read AND written in place.
int sk_mamba2_step(void* x_t, void* B_t, void* C_t, void* A_log_t,
                   void* h_state, void* y_t,
                   uint32_t BH, uint32_t Ds, uint32_t Dv);

int sk_conv1d_silu(void* x, void* weight, void* bias, void* y,
                   uint32_t B, uint32_t L, uint32_t C);

int sk_gate_norm(void* ssm_out, void* z, void* weight, void* y,
                 uint32_t B, uint32_t L, uint32_t E, float eps);

} // extern "C"

#endif
