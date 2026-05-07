#ifndef SK_PAGED_ATTN_H
#define SK_PAGED_ATTN_H
#include <cstdint>
extern "C" {
int sk_paged_attn(void* Q, void* K_cache, void* V_cache, void* block_table,
                  void* seq_lens, void* O, uint32_t num_seqs, uint32_t num_heads,
                  uint32_t head_dim, uint32_t num_kv_heads, uint32_t block_size,
                  uint32_t max_blocks);
}
#endif
