//
//  layernorm.h
//  SuperKittens — host-side norm dispatch
//
//  By Alazar Manakelew

#ifndef SUPERKITTENS_LAYERNORM_H
#define SUPERKITTENS_LAYERNORM_H

namespace meow {
namespace norm {

struct LayerNormParams {
    unsigned int rows;
    unsigned int d;
    float eps = 1e-5f;
};

// Kernels (all 128 threads, grid (1, ceil_div(rows, 4), 1)):
//   layernorm           — y = (x - mean) * rsqrt(var+eps) * gamma + beta     [7 bufs]
//   layernorm_residual  — y = layernorm(x + res, gamma, beta)                 [8 bufs]
//   rmsnorm             — y = x * rsqrt(sumSq/d + eps) * gamma                [6 bufs]
//   rmsnorm_residual    — y = rmsnorm(x + res, gamma)                         [7 bufs]

} // namespace norm
} // namespace meow

#endif // SUPERKITTENS_LAYERNORM_H
