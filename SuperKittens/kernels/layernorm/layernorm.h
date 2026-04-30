//
//  layernorm.h
//  SuperKittens — host-side layernorm dispatch
//
//  By Alazar Manakelew

#ifndef SUPERKITTENS_LAYERNORM_H
#define SUPERKITTENS_LAYERNORM_H

namespace meow {
namespace layernorm {

struct Params {
    unsigned int rows;
    unsigned int d;
    float eps = 1e-5f;
};

// Grid: (1, (rows + 3) / 4, 1), 128 threads.
// Buffers: x, gamma, beta, y, rows, d, eps (7 buffers).

} // namespace layernorm
} // namespace meow

#endif // SUPERKITTENS_LAYERNORM_H
