//
//  conv1d_silu.h
//  SuperKittens — host-side conv1d dispatch
//

#ifndef SUPERKITTENS_CONV1D_SILU_H
#define SUPERKITTENS_CONV1D_SILU_H

namespace meow {
namespace conv1d {

struct Params {
    unsigned int batch;
    unsigned int length;
    unsigned int channels;
    // kernel_size is fixed at 4 (Mamba2 standard)
};

// Grid: (batch, (length + 3) / 4, 1), 128 threads.
// Buffers: x, weight, bias, y, batch, length, channels (7 buffers)
// Weight: (channels, 4), Bias: (channels,)

} // namespace conv1d
} // namespace meow

#endif
