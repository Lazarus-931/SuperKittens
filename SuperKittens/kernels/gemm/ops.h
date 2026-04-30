//
//  ops.h
//  SuperKittens
//
//  GEMM helper ops used by reference and future TK-style kernels.
//

#ifndef SUPERKITTENS_GEMM_OPS_H
#define SUPERKITTENS_GEMM_OPS_H

#include "gemm_impl.h"

namespace meow {
namespace gemm {

#ifdef __METAL_VERSION__

METAL_FUNC uint ceil_div_u32(uint x, uint y) {
    return (x + y - 1) / y;
}

METAL_FUNC float apply_alpha_beta(float acc, float prior, float alpha, float beta) {
    return alpha * acc + beta * prior;
}

METAL_FUNC float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

METAL_FUNC float sigmoid(float x) {
    return 1.0f / (1.0f + metal::fast::exp(-x));
}

METAL_FUNC float silu(float x) {
    return x * sigmoid(x);
}

METAL_FUNC float apply_epilogue(float x, uint32_t epilogue) {
    switch (epilogue) {
        case GEMM_EPILOGUE_BIAS_RELU:
            return relu(x);
        case GEMM_EPILOGUE_BIAS_SILU:
            return silu(x);
        case GEMM_EPILOGUE_BIAS:
        case GEMM_EPILOGUE_NONE:
        default:
            return x;
    }
}

#endif

} // namespace gemm
} // namespace meow

#endif // SUPERKITTENS_GEMM_OPS_H
