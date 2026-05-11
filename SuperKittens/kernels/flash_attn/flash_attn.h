#ifndef SK_FLASH_ATTN_H
#define SK_FLASH_ATTN_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ds4-derived MLA-shaped flash attention. Decode-optimized vec kernel.
// SK's `mha_causal` (kernels/attn/) is faster for d≤128 prefill / multi-query;
// this lives in its own dir as the second flash-attn variant — the right tool
// when dk = dv = 512 (DeepSeek MLA absorption shape).
//
//   Q     : (B, H,    S_q,  dk) fp32   (kernel reinterprets as float4*)
//   K     : (B, H_kv, S_kv, dk) fp16
//   V     : (B, H_kv, S_kv, dv) fp16
//   mask  : (S_q, S_kv)         fp16 — NULL if has_mask=0; supply -inf for
//                                       masked positions, 0 elsewhere
//   O     : (B, S_q, H,    dv)  fp32
//
// (dk, dv) instantiations currently in libsk.metallib: (128,128), (512,512).
// Adding a new (dk, dv) requires a one-line template instantiation in
// flash_attn.metal at the bottom.
int sk_flash_attn_ext_vec(
    void* Q, void* K, void* V, void* mask, void* O,
    uint32_t B, uint32_t H, uint32_t H_kv,
    uint32_t S_q, uint32_t S_kv,
    uint32_t D_k, uint32_t D_v,
    int has_mask, float scale,
    int32_t nsg, int32_t nwg);

#ifdef __cplusplus
}
#endif
#endif
