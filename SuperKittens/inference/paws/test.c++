#include "paws.h"
#include <cstdio>
#include <cassert>

int main() {
    // 1MB heap, 256-byte blocks, max 64 blocks
    paws_ctx* ctx = paws_create(1 << 20, 256, 64);
    printf("free: %u\n", paws_num_free(ctx));

    // alloc 10 blocks
    int32_t blocks[10];
    for (int i = 0; i < 10; i++) {
        blocks[i] = paws_alloc(ctx);
        assert(blocks[i] >= 0);
    }
    printf("after 10 allocs: free=%u used=%u\n", paws_num_free(ctx), paws_num_used(ctx));

    // write to each block
    for (int i = 0; i < 10; i++) {
        uint32_t* ptr = (uint32_t*)paws_block_ptr(ctx, blocks[i]);
        *ptr = i * 100;
    }
    // read back
    for (int i = 0; i < 10; i++) {
        uint32_t* ptr = (uint32_t*)paws_block_ptr(ctx, blocks[i]);
        assert(*ptr == (uint32_t)(i * 100));
    }
    printf("write/read OK\n");

    // free odd ones
    for (int i = 0; i < 10; i += 2) paws_free(ctx, blocks[i]);
    printf("after free 5: free=%u\n", paws_num_free(ctx));

    // re-alloc — should reuse freed blocks
    int32_t reuse[5];
    for (int i = 0; i < 5; i++) {
        reuse[i] = paws_alloc(ctx);
        assert(reuse[i] >= 0);
    }
    printf("re-alloc 5: free=%u\n", paws_num_free(ctx));

    // sequence test
    paws_seq_reserve(ctx, 0, 8);
    for (int i = 0; i < 8; i++) paws_seq_append(ctx, 0);
    printf("seq 0: len=%u blocks=[", paws_seq_len(ctx, 0));
    auto* bt = paws_seq_blocks(ctx, 0);
    for (uint32_t i = 0; i < paws_seq_len(ctx, 0); i++)
        printf("%d%s", bt[i], i < paws_seq_len(ctx, 0)-1 ? " " : "");
    printf("]\n");

    paws_seq_free(ctx, 0);
    printf("after seq free: free=%u used=%u\n", paws_num_free(ctx), paws_num_used(ctx));

    paws_destroy(ctx);
    printf("PASS\n");
    return 0;
}
