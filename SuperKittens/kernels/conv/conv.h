//
//  conv.h — C bindings for conv1d, conv2d, conv3d
//

#ifndef SK_CONV_H
#define SK_CONV_H

#include <cstdint>

extern "C" {

int sk_conv1d(void* x, void* weight, void* bias, void* y,
              uint32_t B, uint32_t L, uint32_t C, uint32_t K);

} // extern "C"

#endif
