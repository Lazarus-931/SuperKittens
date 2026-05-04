#include "paws.h"
#include <cstdio>
#include <cassert>

int main() {
    // Llama 7B: 32 KV heads, 128 head dim, fp16, 16 tokens/block, 1000 blocks
    paws_config cfg = {
        .block_tokens   = 16,
        .num_kv_heads   = 32,
        .head_dim       = 128,
        .precision_bytes = 2,
        .max_blocks     = 1000,
    };

    paws_ctx* ctx = paws_create(&cfg);
    assert(ctx);
    printf("config: %u tokens/block  %u heads  d=%u  %u bytes/block  %u blocks\n",
           paws_block_tokens(ctx), cfg.num_kv_heads, cfg.head_dim,
           paws_block_bytes(ctx), cfg.max_blocks);
    printf("  block_bytes=%u  elements=%u\n", paws_block_bytes(ctx), paws_block_elements(ctx));
    assert(paws_block_bytes(ctx) == 16 * 32 * 128 * 2);  // 131072 bytes = 128KB
    printf("  free=%u\n", paws_num_free(ctx));

    // Simulate 4 sequences, each with different lengths
    uint32_t seqs[] = {0, 1, 2, 3};
    uint32_t tokens[] = {128, 256, 64, 512};  // 8, 16, 4, 32 blocks

    for (int i = 0; i < 4; i++) {
        paws_seq_init(ctx, seqs[i]);
        uint32_t num_blocks = (tokens[i] + cfg.block_tokens - 1) / cfg.block_tokens;
        for (uint32_t j = 0; j < num_blocks; j++) {
            int32_t blk = paws_seq_append(ctx, seqs[i]);
            assert(blk >= 0);
        }
        printf("seq %u: %u tokens → %u blocks (len=%u)\n",
               seqs[i], tokens[i], num_blocks, paws_seq_len(ctx, seqs[i]));

        // Verify block table
        auto* bt = paws_seq_blocks(ctx, seqs[i]);
        assert(bt);
        for (uint32_t j = 0; j < paws_seq_len(ctx, seqs[i]); j++)
            assert(bt[j] >= 0);
    }

    printf("used=%u free=%u\n", paws_num_used(ctx), paws_num_free(ctx));

    // Free seq 1 and 3
    paws_seq_free(ctx, 1);
    paws_seq_free(ctx, 3);
    printf("after free seq 1,3: used=%u free=%u\n", paws_num_used(ctx), paws_num_free(ctx));

    // Re-allocate should reuse
    paws_seq_init(ctx, 5);
    for (int i = 0; i < 8; i++) paws_seq_append(ctx, 5);
    printf("seq 5: len=%u  used=%u\n", paws_seq_len(ctx, 5), paws_num_used(ctx));

    paws_destroy(ctx);

    // Mamba edge case: num_kv_heads=0
    paws_config mamba_cfg = {.block_tokens=16, .num_kv_heads=0, .head_dim=128};
    assert(paws_create(&mamba_cfg) == nullptr);
    printf("Mamba (no KV cache): correctly returns nullptr\n");

    printf("PASS\n");
    return 0;
}
