// TokenArgs — per-token scalar block patched into one shared MTL::Buffer.
//
// WHY: MTLIndirectCommandBuffer forbids setBytes. Per-token scalars
// (current_pos, kv_idx_base, token_id, layer_idx) used to be re-set via
// computeCommandEncoder::setBytes(...) inside the per-token loop; that path
// can't be ICB-recorded. Host writes this 32-byte block via memcpy into a
// pre-bound MTL::Buffer; kernels read it as `constant TokenArgs& [[buffer(N)]]`.

#ifndef SK_INFERENCE_ICB_TOKEN_ARGS_H
#define SK_INFERENCE_ICB_TOKEN_ARGS_H

#include <cstdint>
#include <cstring>

namespace sk {

struct TokenArgs {
    uint32_t current_pos;    // absolute token index in the sequence
    uint32_t kv_idx_base;    // logical_first % cache_max (ring-buffer base)
    int32_t  token_id;       // newest input token id (decode: 1 token / step)
    uint32_t layer_idx;      // currently-dispatched layer (0..n_layers-1)
    uint32_t reserved[4];    // pad to 32 B; future expansion (do not remove)
};

static_assert(sizeof(TokenArgs) == 32, "TokenArgs must be exactly 32 bytes");

// Host-side patcher. `dst` is the .contents() pointer of the MTL::Buffer
// bound at the kernel's TokenArgs slot. No encoder used — write-through to
// the shared-storage buffer is GPU-visible on the next executeCommandsInBuffer.
inline void token_args_patch(void* dst, const TokenArgs& a) {
    std::memcpy(dst, &a, sizeof(TokenArgs));
}

}  // namespace sk

#endif  // SK_INFERENCE_ICB_TOKEN_ARGS_H
