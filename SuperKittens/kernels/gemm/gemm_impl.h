//
//  gemm_impl.h
//  SuperKittens
//
//  Shared GEMM parameter structs and specialization ids.
//  Included from both .metal and .cpp files.
//

#ifndef SUPERKITTENS_GEMM_IMPL_H
#define SUPERKITTENS_GEMM_IMPL_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using stride_t = ulong;
#else
#include <cstdint>
using stride_t = uint64_t;
#endif

namespace meow {
namespace gemm {

enum GemmOp : uint32_t {
    GEMM_OP_N = 0,
    GEMM_OP_T = 1,
};

enum GemmSpecialization : uint32_t {
    GEMM_SPEC_GENERIC = 0,
    GEMM_SPEC_2048_3072_4096 = 1,
    GEMM_SPEC_3072_2048_4096 = 2,
    GEMM_SPEC_4096_4096_4096 = 3,
};

enum GemmEpilogue : uint32_t {
    GEMM_EPILOGUE_NONE = 0,
    GEMM_EPILOGUE_BIAS = 1,
    GEMM_EPILOGUE_BIAS_RELU = 2,
    GEMM_EPILOGUE_BIAS_SILU = 3,
};

struct GemmParams {
    int m;
    int n;
    int k;

    stride_t lda;
    stride_t ldb;
    stride_t ldc;

    float alpha;
    float beta;

    uint32_t op_a;
    uint32_t op_b;
    uint32_t specialization;
    uint32_t epilogue;
};

} // namespace gemm
} // namespace meow

#endif // SUPERKITTENS_GEMM_IMPL_H
