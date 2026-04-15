//
//  index.h
//  SuperKittens
//
//  Mamba-specific tensor layout and indexing helpers.
//

#ifndef SUPERKITTENS_MAMBA_INDEX_H
#define SUPERKITTENS_MAMBA_INDEX_H

#include <metal_stdlib>
using namespace metal;

namespace meow {
namespace mamba {
namespace index {

METAL_FUNC static inline uint ssd_best_cb_index(constant SSDChunkScanArgs &args, uint b, uint c, uint h, uint m, uint k) {
    return (((((b * args.nChunks + c) * args.heads + h) * args.chunkSize + m) * args.chunkSize) + k);
}

METAL_FUNC static inline uint ssd_x_index(constant SSDChunkScanArgs &args, uint b, uint c, uint h, uint m, uint n) {
    return ((((b * args.nChunks + c) * args.chunkSize + m) * args.heads + h) * args.hdim + n);
}

METAL_FUNC static inline uint ssd_out_index(constant SSDChunkScanArgs &args, uint b, uint c, uint h, uint m, uint n) {
    return ssd_x_index(args, b, c, h, m, n);
}

METAL_FUNC static inline uint ssd_vec_index(constant SSDChunkScanArgs &args, uint b, uint c, uint h, uint m) {
    return (((b * args.nChunks + c) * args.heads + h) * args.chunkSize + m);
}

METAL_FUNC static inline uint ssd_cb_index(constant SSDChunkScanArgs &args, uint b, uint c, uint h, uint m, uint k) {
    return (((((b * args.nChunks + c) * args.heads + h) * args.chunkSize + m) * args.chunkSize) + k);
}

METAL_FUNC static inline uint ssd_c_index(constant SSDChunkScanArgs &args, uint b, uint c, uint h, uint m, uint ds) {
    return ((((b * args.nChunks + c) * args.chunkSize + m) * args.heads + h) * args.dstate + ds);
}

METAL_FUNC static inline uint ssd_prev_index(constant SSDChunkScanArgs &args, uint b, uint c, uint h, uint n, uint ds) {
    return (((((b * args.nChunks + c) * args.heads + h) * args.hdim + n) * args.dstate) + ds);
}

} // namespace index
} // namespace mamba
} // namespace meow

#endif // SUPERKITTENS_MAMBA_INDEX_H
