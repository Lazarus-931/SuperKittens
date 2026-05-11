#ifndef SK_FUSION_KV_UP_PAIR_H
#define SK_FUSION_KV_UP_PAIR_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// MLA K-up / V-up dual matmul fusion (fp16, decode-friendly).
//   c_kv    : (T, R)        fp16  — compressed KV cache rows (R = kv_lora_rank)
//   w_k_up  : (R, K_OUT)    fp16  — K up-projection,   K_OUT = n_heads * qk_nope_dim
//   w_v_up  : (R, V_OUT)    fp16  — V up-projection,   V_OUT = n_heads * v_head_dim
// Outputs:
//   k_no_pe : (T, K_OUT)    fp16
//   v_out   : (T, V_OUT)    fp16
//
// Tile: 256 threads / 8 simdgroups. One TG covers 16 output cols. Grid x-axis
// partitioned: first K_OUT/16 TGs write K, next V_OUT/16 write V. Optimized for
// T=1 (decode). At T≥2 the matvec design loses to two GEMMs — caller should
// fall back to plain gemm_fp16 in that regime (see results.md in this dir).
int sk_kv_up_pair(void* c_kv, void* w_k_up, void* w_v_up,
                  void* k_no_pe, void* v_out,
                  uint32_t T, uint32_t R, uint32_t K_OUT, uint32_t V_OUT);

#ifdef __cplusplus
}
#endif
#endif
