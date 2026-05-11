//
//  add.metal — elementwise y = a + b.  Macro-instantiated for {f16, f32, bf16}.
//  vec4 path covers n/4 elems; scalar tail handles the remainder.
//

#include <metal_stdlib>
using namespace metal;

namespace meow::ops::add {

// ─── add_<dtype> with half4 / float4 / bfloat4 vectorized loop ────────
// Each thread processes 4 scalar elements at a time when possible.
//
// Uniform parameter across all kernels: `n` is the total number of scalars
// (not vec4 chunks). The kernel computes vec_idx and a final tail for any
// trailing 1-3 scalars.

#define DEFINE_ADD_KERNEL(NAME, T, T4)                                      \
[[host_name(#NAME)]]                                                        \
[[kernel]]                                                                  \
void NAME(                                                                  \
    device const T*  a   [[buffer(0)]],                                     \
    device const T*  b   [[buffer(1)]],                                     \
    device       T*  y   [[buffer(2)]],                                     \
    constant uint& n     [[buffer(3)]],                                     \
    uint gid             [[thread_position_in_grid]])                       \
{                                                                           \
    /* vec4 path covers floor(n/4) elements */                              \
    const uint n4 = n / 4u;                                                 \
    if (gid < n4) {                                                         \
        T4 va = ((device const T4*)a)[gid];                                 \
        T4 vb = ((device const T4*)b)[gid];                                 \
        ((device T4*)y)[gid] = va + vb;                                     \
        return;                                                             \
    }                                                                       \
    /* scalar tail: gid in [n4, n4 + (n & 3)) — at most 3 stragglers */     \
    const uint tail_idx = (gid - n4) + n4 * 4u;                             \
    if (tail_idx < n) {                                                     \
        y[tail_idx] = a[tail_idx] + b[tail_idx];                            \
    }                                                                       \
}

DEFINE_ADD_KERNEL(add_f16,  half,   half4)
DEFINE_ADD_KERNEL(add_f32,  float,  float4)
DEFINE_ADD_KERNEL(add_bf16, bfloat, bfloat4)

#undef DEFINE_ADD_KERNEL

} // namespace meow::ops::add
