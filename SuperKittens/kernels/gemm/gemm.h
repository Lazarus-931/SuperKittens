//
//  gemm.h — C bindings for GEMM (fp16 + fp8)
//

#ifndef SK_GEMM_H
#define SK_GEMM_H

#include <cstdint>

extern "C" {

int sk_gemm_fp16(void* A, void* B, void* C, void* bias,
                 uint32_t M, uint32_t N, uint32_t K,
                 uint32_t ldA, uint32_t ldB, uint32_t ldC,
                 int transA, int transB, int has_bias);

int sk_gemm_fp8(void* A, void* B, void* C, void* bias,
                uint32_t M, uint32_t N, uint32_t K,
                uint32_t ldA, uint32_t ldB, uint32_t ldC,
                int transA, int transB, int has_bias);

} // extern "C"

#endif
