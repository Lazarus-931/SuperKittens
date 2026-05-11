#ifndef SK_DEEPSEEK_ROPE_TAIL_H
#define SK_DEEPSEEK_ROPE_TAIL_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// DeepSeek V4 partial p-RoPE on the last `n_dims` elements of each row.
//   x         : (ne03, ne02, ne01, ne00)   fp32   row-major contiguous
//   pos       : (ne02,)                    int32   token positions
//   freq      : (n_dims/2,)                fp32    optional freq factors (NULL → 1.0 each)
//   out       : same shape as x
// Computes: pass-through for the first (ne00 - n_dims) elements, YaRN p-RoPE
// rotation on the trailing n_dims (interleaved or NEOX). For DS4: mode=2 (NEOX).
int sk_deepseek_rope_tail(
    void* x, void* pos, void* freq, void* out,
    uint32_t ne03, uint32_t ne02, uint32_t ne01, uint32_t ne00,
    int32_t  n_dims, int32_t mode, int32_t n_ctx_orig,
    float    freq_base, float freq_scale, float ext_factor,
    float    attn_factor, float beta_fast, float beta_slow,
    int32_t  inverse);

#ifdef __cplusplus
}
#endif
#endif
