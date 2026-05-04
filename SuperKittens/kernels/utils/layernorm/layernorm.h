//
//  layernorm.h — C bindings for LayerNorm, RMSNorm
//

#ifndef SK_LAYERNORM_H
#define SK_LAYERNORM_H

#include <cstdint>

extern "C" {

int sk_layernorm(void* x, void* gamma, void* beta, void* y,
                 uint32_t rows, uint32_t d, float eps);

int sk_rmsnorm(void* x, void* weight, void* y,
               uint32_t rows, uint32_t d, float eps);

} // extern "C"

#endif
