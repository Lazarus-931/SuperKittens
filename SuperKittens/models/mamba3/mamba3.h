//
//  mamba3.h — C bindings for Mamba-3 kernels
//

#ifndef SK_MAMBA3_H
#define SK_MAMBA3_H

#include <cstdint>

extern "C" {

int sk_mamba3_pre_ssm(void* xBC, void* dt, void* angle, void* norm_w,
                      void* Q_out, void* K_out, void* V_out,
                      void* A_out, void* B_out,
                      uint32_t BH, uint32_t L, uint32_t DQ, float eps);

int sk_mamba3_ssm(void* Q, void* K, void* V, void* A, void* B, void* angle,
                  void* O, uint32_t BH, uint32_t L, uint32_t DQ, uint32_t DV, uint32_t CS);

// Variant with optional persistent SSM state.
//   h_state: (BH, DQ, DV) fp32   a_cs:    (BH,) fp32 — running cumsum of A.
// Pass NULL to a *_in to start fresh, NULL to *_out to discard.
int sk_mamba3_ssm_state(void* Q, void* K, void* V, void* A, void* B, void* angle, void* O,
                        void* h_state_in, void* h_state_out,
                        void* a_cs_in,    void* a_cs_out,
                        uint32_t BH, uint32_t L, uint32_t DQ, uint32_t DV, uint32_t CS);

// Single-token decode step. Per-token shapes:
//   q_t,k_t: (BH, DQ) fp16     v_t: (BH, DV) fp16
//   a_t,b_t: (BH,)    fp16     angle_t: (BH, DQ/2) fp16
//   y_t:     (BH, DV) fp16
//   h_state: (BH, DQ, DV) fp32 — read & written in-place
//   a_cs:    (BH,) fp32       — read & written in-place (running cumsum)
int sk_mamba3_step(void* q_t, void* k_t, void* v_t, void* a_t, void* b_t, void* angle_t,
                   void* h_state, void* a_cs, void* y_t,
                   uint32_t BH, uint32_t DQ, uint32_t DV);

int sk_mamba3_post_ssm(void* z, void* ssm_out, void* norm_w, void* gated,
                       uint32_t BH, uint32_t L, uint32_t DV, float eps);

} // extern "C"

#endif
