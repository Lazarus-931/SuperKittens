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

int sk_mamba3_post_ssm(void* z, void* ssm_out, void* norm_w, void* gated,
                       uint32_t BH, uint32_t L, uint32_t DV, float eps);

} // extern "C"

#endif
