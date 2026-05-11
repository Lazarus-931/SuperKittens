//
//  router.h — MoE router C bindings
//
#ifndef SK_MOE_ROUTER_H
#define SK_MOE_ROUTER_H
#include <cstdint>
extern "C" {
// x: (T,D) fp16; W: (D,N) fp16; top_idx: (T,K) int32; top_score: (T,K) fp16
int sk_moe_router(void* x, void* W, void* top_idx, void* top_score,
                  uint32_t T, uint32_t D, uint32_t N, uint32_t K);
}
#endif
