#ifndef SK_ATTN_H
#define SK_ATTN_H
#include <cstdint>
extern "C" {
int sk_attn(void* Q, void* K, void* V, void* O, uint32_t seq, uint32_t head_dim, uint32_t nheads, int causal);
}
#endif
