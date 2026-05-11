//
//  embedding.h — C bindings for token-id embedding lookup
//

#ifndef SK_EMBEDDING_H
#define SK_EMBEDDING_H

#include <cstdint>

extern "C" {

// Gather rows from `table` (V, D) by indices `ids` (N,) into `out` (N, D).
// half precision table+output, int32 ids. D must be % 4.
// Out-of-range ids zero-fill the output row.
int sk_embedding_lookup(void* table, void* ids, void* out,
                        uint32_t N, uint32_t D, uint32_t V);

} // extern "C"

#endif
