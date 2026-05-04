//
//  activation.h — C bindings for GELU, SiLU, ReLU
//  Compiled into libsk.dylib.  Called from activation.py via ctypes.
//

#ifndef SK_ACTIVATION_H
#define SK_ACTIVATION_H

#include <cstdint>

extern "C" {

// All three share the same signature: (x, y, rows, cols)
// Grid: (1, (rows+3)/4, 1), threads: (128, 1, 1)

int sk_activation(const char* kernel_name,
                  void* x, void* y,
                  uint32_t rows, uint32_t cols);

} // extern "C"

#endif
