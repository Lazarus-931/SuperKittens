//
//  rotary.h — C bindings for RoPE
//

#ifndef SK_ROTARY_H
#define SK_ROTARY_H

#include <cstdint>

extern "C" {

int sk_rope(void* Q, void* K, void* cos, void* sin,
            uint32_t seq, uint32_t head_dim, uint32_t n_heads);

} // extern "C"

#endif
